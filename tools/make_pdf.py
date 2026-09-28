"""Build the report PDF: markdown -> print-styled HTML -> Chrome's print engine.

    python tools/make_pdf.py

Chrome is used because it is the one PDF engine present on a typical Windows machine - no pandoc,
no LibreOffice, no LaTeX needed. It is also how the original plan-of-action PDF was produced (its
metadata names Skia/PDF, Chrome's rasteriser).

Override paths if needed:
    python tools/make_pdf.py docs/report.md docs/ISP_Network_Planner_Report.pdf
"""
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import md2html  # noqa: E402  (same directory)

CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
]


def find_browser():
    for p in CANDIDATES:
        if os.path.isfile(p):
            return p
    raise SystemExit(
        "No Chrome or Edge found. Install one, or open docs/report.md's HTML in any browser and "
        "print to PDF manually (the HTML is written next to the PDF when this fails).")


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "docs", "report.md")
    dst = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        ROOT, "docs", "ISP_Network_Planner_Report.pdf")
    src, dst = os.path.abspath(src), os.path.abspath(dst)

    with open(src, encoding="utf-8") as fh:
        page = md2html.convert(fh.read())
    tmpdir = tempfile.mkdtemp(prefix="isp-report-")
    html_path = os.path.join(tmpdir, "report.html")
    with open(html_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(page)
    print(f"html:  {html_path}  ({len(page):,} bytes)")

    browser = find_browser()
    print(f"engine: {browser}")
    cmd = [
        browser, "--headless", "--disable-gpu", "--no-sandbox",
        "--no-pdf-header-footer",                    # no URL/date stamp on a submitted document
        "--run-all-compositor-stages-before-draw",
        "--virtual-time-budget=10000",               # let fonts and layout settle
        f"--user-data-dir={os.path.join(tmpdir, 'profile')}",
        f"--print-to-pdf={dst}",
        "file:///" + html_path.replace("\\", "/"),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if not os.path.isfile(dst):
        print(r.stdout[-2000:], r.stderr[-2000:], sep="\n")
        raise SystemExit("PDF was not produced")
    print(f"pdf:   {dst}  ({os.path.getsize(dst):,} bytes)")


if __name__ == "__main__":
    main()
