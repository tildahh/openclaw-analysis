"""Export COOKBOOK.md to one self-contained HTML file (and a PDF) with every diagram and image inline.

Headless Chrome renders the Markdown and the Mermaid diagrams, then the finished page is saved with the
scripts removed: the diagrams stay as inline SVG and the images as data URIs, so the HTML opens anywhere,
offline. Rendering needs Google Chrome and network access to cdn.jsdelivr.net; the output needs neither.

    python3 -m scripts.export_cookbook                      # output/COOKBOOK.html and output/COOKBOOK.pdf
    python3 -m scripts.export_cookbook --theme dark --no-pdf
    python3 -m scripts.export_cookbook --no-pdf --artifact output/COOKBOOK.artifact.html   # page for a claude.ai Artifact
"""

import argparse
import base64
import html
import json
import mimetypes
import os
from pathlib import Path
import re
import subprocess
import tempfile

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
MARKED = "https://cdn.jsdelivr.net/npm/marked@12.0.2/marked.min.js"
MERMAID = "https://cdn.jsdelivr.net/npm/mermaid@11.12.2/dist/mermaid.min.js"
IMAGE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)\)")
HTML_IMAGE = re.compile(r'(<img\b[^>]*?\bsrc=")([^"]+)(")')
LOCAL_LINK = re.compile(r'(<a\b[^>]*?\bhref=")(?!https?:|mailto:|#|data:)([^"]+)(")')

THEMES = {
    "light": "--bg:#ffffff;--fg:#1f2328;--muted:#59636e;--line:#d1d9e0;--code:#f6f8fa;--link:#0969da;",
    "dark": "--bg:#1f1f1f;--fg:#e6e6e6;--muted:#a0a7b4;--line:#3c3f45;--code:#2b2d31;--link:#6cb6ff;",
}

CSS = """
:root { %(theme)s }
body { margin: 0; background: var(--bg); color: var(--fg);
  font: 16px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
article { max-width: 920px; margin: 0 auto; padding: 40px 24px 80px; }
h1, h2, h3 { line-height: 1.25; margin: 1.6em 0 0.6em; }
h1 { font-size: 2em; } h2 { font-size: 1.5em; border-bottom: 1px solid var(--line); padding-bottom: .3em; }
a { color: var(--link); }
img { max-width: 100%%; height: auto; display: block; margin: 1em auto; }
.mermaid { margin: 1.2em 0; text-align: center; }
table { border-collapse: collapse; margin: 1em 0; display: block; overflow-x: auto; }
th, td { border: 1px solid var(--line); padding: 6px 12px; vertical-align: top; }
th { background: var(--code); }
code { font: 0.88em ui-monospace, SFMono-Regular, Menlo, monospace; background: var(--code);
  padding: .15em .35em; border-radius: 4px; }
pre { background: var(--code); padding: 14px 16px; border-radius: 6px; overflow-x: auto; }
pre code { background: none; padding: 0; }
blockquote { margin: 1em 0; padding: 0 1em; color: var(--muted); border-left: .25em solid var(--line); }
@page { margin: 12mm; }
@media print {
  article { max-width: none; padding: 0; }
  img { max-width: 88%%; }
  pre { white-space: pre-wrap; font-size: 0.8em; }
  img, svg, pre, blockquote, tr { break-inside: avoid; }  /* long tables may break between rows */
  img { max-height: 80vh; width: auto; }  /* a tall figure shrinks instead of jumping to the next page */
  table { font-size: 0.85em; }
  h2, h3, p:has(+ pre), p:has(+ p > img), p:has(+ table), p:has(+ .mermaid) { break-after: avoid; }
}
"""

ARTIFACT_FONTS = ('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500'
                  '&family=IBM+Plex+Sans+Condensed:wght@600&family=IBM+Plex+Sans:ital,wght@0,400;0,600;1,400&display=swap">')

# Artifact pages follow the viewer's theme: light tokens on :root, dark ones for a dark system setting or toggle.
ARTIFACT_CSS = """
:root { --paper:#f7f8fa; --ink:#1c2230; --muted:#5d6678; --rule:#dfe3ea; --code-bg:#edf0f4; --accent:#c2491d;
  --sans:"IBM Plex Sans", -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
  --cond:"IBM Plex Sans Condensed", "IBM Plex Sans", "Arial Narrow", sans-serif;
  --mono:"IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace; }
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) { --paper:#14161b; --ink:#e5e8ee; --muted:#9aa3b3; --rule:#2a2f38;
    --code-bg:#1d2129; --accent:#ff8a57; color-scheme: dark; } }
:root[data-theme="dark"] { --paper:#14161b; --ink:#e5e8ee; --muted:#9aa3b3; --rule:#2a2f38;
  --code-bg:#1d2129; --accent:#ff8a57; color-scheme: dark; }
body { background: var(--paper); color: var(--ink); font: 16.5px/1.65 var(--sans); }
.page { padding-inline: max(16px, 4vw); padding-block: 40px 96px; }
article { max-width: 62rem; margin-inline: auto; }
article > :is(p, ul, ol, blockquote, h1, h2, h3, pre) { max-width: 42rem; margin-inline: auto; }
article > p:has(> img) { max-width: none; }
.eyebrow { font: 500 0.78rem/1 var(--mono); letter-spacing: 0.08em; text-transform: uppercase; color: var(--accent);
  margin-block: 0 0.6em; }
h1 { font: 600 clamp(1.9rem, 4.2vw, 2.6rem)/1.15 var(--cond); text-wrap: balance; margin-block: 0 0.6em; }
h2 { font: 600 1.55rem/1.2 var(--cond); text-wrap: balance; margin-block: 2.4em 0.7em; padding-top: 1em;
  border-top: 1px solid var(--rule); }
h3 { font: 600 1.1rem/1.3 var(--sans); margin-block: 1.8em 0.4em; }
p, li { margin-block: 0 0.9em; }
a { color: var(--accent); text-underline-offset: 2px; }
a:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
code { font: 0.86em var(--mono); background: var(--code-bg); padding: 0.12em 0.36em; border-radius: 4px; }
pre { background: var(--code-bg); padding: 14px 16px; border-radius: 6px; overflow-x: auto; font-size: 0.9rem;
  line-height: 1.5; }
pre code { background: none; padding: 0; font-size: inherit; }
blockquote { margin-block: 1.2em; padding: 0.1em 0 0.1em 1em; border-left: 3px solid var(--accent); color: var(--muted); }
blockquote p { margin: 0; }
img { display: block; max-width: 100%; height: auto; margin: 1.6em auto 0.6em; border-radius: 8px; }
article > p:has(> em:only-child) { color: var(--muted); font-size: 0.92rem; }
.mermaid { margin-block: 1.6em; overflow-x: auto; text-align: center; }
.scroll { overflow-x: auto; margin-block: 1.4em; }
table { border-collapse: collapse; font-size: 0.92rem; font-variant-numeric: tabular-nums; margin-inline: auto; }
th, td { border-bottom: 1px solid var(--rule); padding: 8px 12px; vertical-align: top; }
th { font-weight: 600; border-bottom-width: 2px; }
th:not([align]) { text-align: left; }
.todo { font: 0.84rem/1.5 var(--mono); color: var(--muted); border: 1px dashed var(--rule); border-radius: 6px;
  padding: 8px 12px; }
"""

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>%(title)s</title><style>%(css)s</style>
<script src="%(marked)s"></script><script src="%(mermaid)s"></script></head>
<body><article id="doc"></article>
<script>
const source = %(source)s;
document.getElementById("doc").innerHTML = marked.parse(source);
for (const code of document.querySelectorAll("code.language-mermaid")) {
  const block = document.createElement("div");
  block.className = "mermaid";
  block.textContent = code.textContent;
  code.parentElement.replaceWith(block);
}
mermaid.initialize({ startOnLoad: false });
mermaid.run({ querySelector: ".mermaid" }).then(
  () => document.body.setAttribute("data-export", "done"),
  (error) => document.body.setAttribute("data-export", "error: " + error));
</script></body></html>"""


def inline_images(markdown, base):
    """Replace local images, written as ![alt](src) or <img src>, with base64 data URIs so the export needs no other files."""
    def encode_local_image(src):
        """Return a data URI for a local image file, or None for a remote or missing one."""
        path = (base / src).resolve()
        if src.startswith(("http://", "https://", "data:")) or not path.is_file():
            return None
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"

    def replace_markdown_image(match):
        """Embed the source of one ![alt](src) image."""
        alt, src = match.groups()
        uri = encode_local_image(src)
        return f"![{alt}]({uri})" if uri else match.group(0)

    def replace_html_image(match):
        """Embed the source of one <img> tag, keeping its other attributes such as width."""
        uri = encode_local_image(match.group(2))
        return match.group(1) + uri + match.group(3) if uri else match.group(0)

    return HTML_IMAGE.sub(replace_html_image, IMAGE.sub(replace_markdown_image, markdown))


def render_page(markdown, title, theme):
    """Render the Markdown and diagrams in headless Chrome and return the finished DOM as HTML."""
    page = PAGE % {"title": title, "css": CSS % {"theme": THEMES[theme]}, "marked": MARKED, "mermaid": MERMAID,
                   "source": json.dumps(markdown).replace("</", "<\\/")}
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "page.html"
        source.write_text(page)
        result = subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--virtual-time-budget=30000",
                                 "--dump-dom", source.as_uri()], capture_output=True, text=True, check=True)
    return result.stdout


def check_rendered(dom, expected_diagrams):
    """Fail loudly unless every diagram rendered and the page finished without errors."""
    status = re.search(r'data-export="([^"]*)"', dom)
    if not status or status.group(1) != "done":
        raise SystemExit(f"Export did not finish: {status.group(1) if status else 'no status (timed out?)'}")
    rendered = len(re.findall(r"<svg[^>]*aria-roledescription=", dom))
    if rendered != expected_diagrams or "Syntax error in text" in dom:
        raise SystemExit(f"Rendered {rendered} of {expected_diagrams} diagrams; check the Mermaid blocks.")


def rebase_links(dom, source_dir, out_dir):
    """Point relative links at the same files from the output folder, so evidence links work in the saved HTML and PDF."""
    prefix = Path(os.path.relpath(source_dir.resolve(), out_dir.resolve()))
    return LOCAL_LINK.sub(lambda match: match.group(1) + (prefix / match.group(2)).as_posix() + match.group(3), dom)


def strip_scripts(dom):
    """Remove script tags so the saved page is static and needs no network."""
    return re.sub(r"<script\b[^>]*>.*?</script>", "", dom, flags=re.S)


def write_artifact(dom, path, title):
    """Write the rendered cookbook as an Artifact page: a title, fonts, theme-aware styles and the article."""
    body = re.search(r'<article id="doc">(.*)</article>', dom, re.S).group(1)
    body = re.sub(r"(<table>.*?</table>)", r'<div class="scroll">\1</div>', body, flags=re.S)
    body = body.replace("<p>TODO(Matilda)", '<p class="todo">TODO(Matilda)')
    body = body.replace("<h1>", '<p class="eyebrow">Phoenix cookbook</p>\n<h1>', 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"<title>{html.escape(title)}</title>\n{ARTIFACT_FONTS}\n<style>{ARTIFACT_CSS}</style>\n"
                    f'<main class="page"><article>{body}</article></main>\n')


def main():
    """Export the cookbook to HTML, and to PDF unless --no-pdf is given."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1] / "COOKBOOK.md")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[1] / "output")
    parser.add_argument("--theme", choices=sorted(THEMES), default="light")
    parser.add_argument("--no-pdf", action="store_true", help="write only the HTML file")
    parser.add_argument("--artifact", type=Path, help="also write a page ready to publish as a claude.ai Artifact")
    parser.add_argument("--artifact-title", default="OpenClaw Heartbeat Cookbook", help="the Artifact's short name")
    args = parser.parse_args()

    markdown = args.source.read_text()
    title = next((line[2:].strip() for line in markdown.splitlines() if line.startswith("# ")), args.source.stem)
    dom = render_page(inline_images(markdown, args.source.parent), title, args.theme)
    check_rendered(dom, markdown.count("```mermaid"))
    if args.artifact:
        write_artifact(dom, args.artifact, args.artifact_title)
        print(f"wrote {args.artifact} ({args.artifact.stat().st_size // 1024} KB)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    html_path = args.out_dir / f"{args.source.stem}.html"
    html_path.write_text("<!doctype html>\n" + rebase_links(strip_scripts(dom), args.source.parent, args.out_dir))
    print(f"wrote {html_path} ({html_path.stat().st_size // 1024} KB)")
    if not args.no_pdf:
        pdf_path = args.out_dir / f"{args.source.stem}.pdf"
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={pdf_path}", html_path.resolve().as_uri()], capture_output=True, check=True)
        print(f"wrote {pdf_path} ({pdf_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
