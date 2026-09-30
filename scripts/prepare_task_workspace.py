"""Prepare local, hash-verified workspace seeds and new independent task copies."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


DEFAULT_EXCLUSIONS = ("drafts", "briefings", "phoenix-eval-output", ".git",
                      ".openclaw", "sessions", ".sessions", "runtime", ".runtime",
                      "indexes", ".indexes", ".cache")


def resolve_safe_path(value):
    """Resolve a path only after rejecting symbolic links in its existing ancestry."""
    path = Path(value).absolute()
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError(f"Symbolic links are not allowed: {part}")
    return path.resolve()


def require_disjoint(*paths):
    """Reject equal, nested or containing paths before any destination is created."""
    for index, first in enumerate(paths):
        for second in paths[index + 1:]:
            if first.is_relative_to(second) or second.is_relative_to(first):
                raise ValueError(f"Paths must not overlap: {first} and {second}")


def normalize_exclusions(exclusions):
    """Validate explicit exclusions as source-relative files or directory prefixes."""
    result = set(DEFAULT_EXCLUSIONS)
    for value in exclusions:
        path = Path(value)
        if not value or path.is_absolute() or ".." in path.parts or str(path) == ".":
            raise ValueError(f"Exclusion must be a relative path: {value}")
        result.add(path.as_posix())
    return sorted(result)


def hash_file(path):
    """Hash file contents without loading the entire file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory_workspace(source, exclusions=()):
    """Record included and excluded files and empty directories without following links."""
    source = resolve_safe_path(source)
    if not source.is_dir():
        raise ValueError(f"Source is not a directory: {source}")
    copied, excluded, directories = [], [], []
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source).as_posix()
        if path.is_symlink():
            raise ValueError(f"Symbolic links are not allowed: {path}")
        omitted = any(relative == item or relative.startswith(item + "/") for item in exclusions)
        if path.is_dir():
            if not omitted:
                directories.append(relative)
        elif path.is_file():
            item = {"path": relative, "sha256": hash_file(path), "size": path.stat().st_size}
            (excluded if omitted else copied).append(item)
        else:
            raise ValueError(f"Only ordinary files and directories are supported: {path}")
    return {"files": copied, "directories": directories, "excluded_files": excluded}


def hash_inventory(inventory):
    """Identify the actor-visible seed, including paths, sizes, contents and empty directories."""
    content = {key: inventory[key] for key in ("files", "directories")}
    return hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def copy_inventory(source, destination, inventory):
    """Copy only enumerated files into a new private directory and verify their hashes."""
    destination.mkdir(mode=0o700)
    for relative in inventory["directories"]:
        (destination / relative).mkdir(parents=True, exist_ok=True, mode=0o700)
    for item in inventory["files"]:
        source_file = source / item["path"]
        resolve_safe_path(source_file)
        target = destination / item["path"]
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        shutil.copyfile(source_file, target)
        target.chmod(0o600)
    actual = inventory_workspace(destination)
    if hash_inventory(actual) != hash_inventory(inventory):
        raise ValueError("Copied files changed during preparation; retain this failed copy for inspection")


def write_manifest(path, manifest):
    """Write a private audit manifest without overwriting an existing file."""
    with path.open("x") as stream:
        path.chmod(0o600)
        json.dump(manifest, stream, indent=2)
        stream.write("\n")


def create_snapshot(source, snapshot, exclusions=(), dry_run=False):
    """Create a reviewed seed and external inventory without modifying the source workspace."""
    source, snapshot = resolve_safe_path(source), resolve_safe_path(snapshot)
    require_disjoint(source, snapshot)
    if snapshot.exists():
        raise ValueError(f"Snapshot destination already exists: {snapshot}")
    exclusions = normalize_exclusions(exclusions)
    inventory = inventory_workspace(source, exclusions)
    manifest = {"format": "task-workspace-v1", "source": str(source),
                "exclusions": exclusions, **inventory, "seed_sha256": hash_inventory(inventory)}
    if not dry_run:
        snapshot.mkdir(parents=True, mode=0o700)
        copy_inventory(source, snapshot / "workspace", inventory)
        write_manifest(snapshot / "manifest.json", manifest)
    return manifest


def verify_snapshot(snapshot, expected_seed_sha256):
    """Verify the seed against both its manifest and the previously recorded expected hash."""
    snapshot = resolve_safe_path(snapshot)
    manifest_path = resolve_safe_path(snapshot / "manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != "task-workspace-v1":
        raise ValueError("Unsupported workspace manifest")
    actual = inventory_workspace(snapshot / "workspace")
    actual_hash = hash_inventory(actual)
    if not expected_seed_sha256 or actual_hash != expected_seed_sha256 or actual_hash != manifest["seed_sha256"]:
        raise ValueError("Seed hash mismatch; preparation stopped")
    if hash_inventory(manifest) != actual_hash:
        raise ValueError("Seed manifest does not match the workspace")
    return manifest


def clone_snapshot(snapshot, destination, expected_seed_sha256, dry_run=False):
    """Start a new trial from the verified seed, leaving all earlier work and results intact."""
    snapshot, destination = resolve_safe_path(snapshot), resolve_safe_path(destination)
    manifest = verify_snapshot(snapshot, expected_seed_sha256)
    source = resolve_safe_path(manifest["source"])
    receipt = resolve_safe_path(destination.parent / (destination.name + ".manifest.json"))
    require_disjoint(source, snapshot, destination, receipt)
    if destination.exists() or receipt.exists():
        raise ValueError("Trial destination or its manifest already exists; use a new destination")
    record = {"format": "task-workspace-trial-v1", "snapshot": str(snapshot),
              "workspace": str(destination), "seed_sha256": manifest["seed_sha256"],
              "files": manifest["files"], "directories": manifest["directories"]}
    if not dry_run:
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        copy_inventory(snapshot / "workspace", destination, manifest)
        write_manifest(receipt, record)
    return record


def main():
    """Inventory a seed or prepare a fresh trial without accessing remote systems or models."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    snapshot = commands.add_parser("snapshot")
    snapshot.add_argument("source", type=Path)
    snapshot.add_argument("snapshot", type=Path)
    snapshot.add_argument("--exclude", action="append", default=[])
    snapshot.add_argument("--dry-run", action="store_true")
    clone = commands.add_parser("clone")
    clone.add_argument("snapshot", type=Path)
    clone.add_argument("destination", type=Path)
    clone.add_argument("--expected-seed-sha256", required=True)
    clone.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "snapshot":
            result = create_snapshot(args.source, args.snapshot, args.exclude, args.dry_run)
        else:
            result = clone_snapshot(args.snapshot, args.destination, args.expected_seed_sha256, args.dry_run)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, str(error) + "\n")
    print(json.dumps({"dry_run": args.dry_run, **result}, indent=2))


if __name__ == "__main__":
    main()
