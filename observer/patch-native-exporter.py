"""Build the narrowly scoped Phoenix presentation patch for one exact OpenClaw exporter."""
import argparse
import hashlib
import json
from pathlib import Path

UPSTREAM_SHA256 = "f0366f582292f8744bab8ae4e428f2e92d00e542f9fd9753a573105fe4e5cf21"


def replace_once(source, old, new):
    """Require a unique upstream anchor before replacing it."""
    if source.count(old) != 1:
        raise ValueError(f"Expected one patch anchor, found {source.count(old)}: {old[:80]}")
    return source.replace(old, new, 1)


def build_patch(source, observer_dir):
    """Enrich existing native spans without changing their IDs or assistant behavior."""
    if hashlib.sha256(source.encode()).hexdigest() != UPSTREAM_SHA256:
        raise ValueError("Native exporter version/hash changed; inspect upstream before patching.")
    helper = (Path(observer_dir) / "live-trace.js").as_uri()
    source = f'import {{ buildSessionAttributes, buildRunAttributes, captureModelOutput, markAggregateUsage, addToolResultNames }} from {json.dumps(helper)};\n' + source
    for function, following in [("recordHarnessRunCompleted", "recordHarnessRunError"), ("recordHarnessRunError", "recordContextAssembled")]:
        start = source.index(f"\tconst {function} =")
        end = source.index(f"\tconst {following} =", start)
        section = source[start:end]
        anchor = "\t\tconst trustedTrace = trustedTraceContext(evt, metadata);"
        section = replace_once(section, anchor, '\t\tconst phoenixRoot = buildRunAttributes(evt, runtime.contentCapturePolicy);\n\t\tObject.assign(spanAttrs, phoenixRoot.attrs);\n' + anchor)
        section = replace_once(section, '\t\tsetSpanAttrs(span, spanAttrs);', '\t\tsetSpanAttrs(span, spanAttrs);\n\t\tif (phoenixRoot.name) span.updateName(phoenixRoot.name);')
        source = source[:start] + section + source[end:]
    source = replace_once(source, "\tconst addRunAttrs = (spanAttrs, evt) => {", "\tconst addRunAttrs = (spanAttrs, evt) => {\n\t\tObject.assign(spanAttrs, buildSessionAttributes(evt));")
    anchor = "\t\tassignOtelModelContentAttributes(spanAttrs, modelContent, contentCapturePolicy);"
    source = replace_once(source, anchor, anchor + "\n\t\taddToolResultNames(spanAttrs, evt, modelContent, contentCapturePolicy);\n\t\tconst phoenixView = captureModelOutput(evt, modelContent, contentCapturePolicy);\n\t\tObject.assign(spanAttrs, phoenixView.attrs);")
    anchor = "\t\taddUpstreamRequestIdSpanEvent(span, evt.upstreamRequestIdHash);"
    child = '''\t\tif (phoenixView.reasoningText) spanWithDuration("Recorded reasoning", {
\t\t\t...phoenixView.attrs,
\t\t\t"openinference.span.kind": "CHAIN",
\t\t\t"input.value": "Recorded reasoning returned by this model call; not a complete account of internal computation.",
\t\t\t"input.mime_type": "text/plain",
\t\t\t"output.value": phoenixView.reasoningText,
\t\t\t"output.mime_type": "text/plain",
\t\t\t"openclaw.presentation.observation_only": true
\t\t}, 0, { parentContext: trace.setSpan(ROOT_CONTEXT, span), endTimeMs: evt.ts }).end(evt.ts);
'''
    source = replace_once(source, anchor, child + anchor)
    anchor = '\t\tspanWithDuration("openclaw.model.usage", spanAttrs, evt.durationMs, {'
    source = replace_once(source, anchor, '\t\tmarkAggregateUsage(spanAttrs, evt);\n' + anchor)
    return source


def main():
    """Write a reviewable patched copy and print only its source hashes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--observer-dir", type=Path, required=True)
    args = parser.parse_args()
    patched = build_patch(args.source.read_text(), args.observer_dir)
    args.output.write_text(patched)
    print(json.dumps({"upstream_sha256": UPSTREAM_SHA256, "patched_sha256": hashlib.sha256(patched.encode()).hexdigest()}))


if __name__ == "__main__":
    main()
