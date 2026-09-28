"""Markdown -> print-ready HTML, for producing the report PDF via Chrome's print engine.

    python tools/md2html.py docs/report.md build/report.html

Then Chrome renders it (see tools/make_pdf.sh, or the command in the README). This is the same
route the original plan-of-action PDF was produced by - its metadata names Skia/PDF, Chrome's
rasteriser.

Handles the subset of Markdown the report uses: h1-h4, paragraphs, **bold**, *italic*, `code`,
[links], tables with \\| escapes, fenced code blocks, bullet and numbered lists, --- rules.
"""
import html
import re
import sys

# Backslash-escapes are stripped in prose, headings and table cells, but never inside code -
# the Appendix's `venv\Scripts\activate` must survive.
_ESCAPED = re.compile(r"\\([\\`*_{}\[\]()#+\-.!$|])")
_INLINE = re.compile(
    r"(`[^`]+`)"                        # code span
    r"|(\*\*(?:\\\*|[^*])+\*\*)"        # bold, tolerating an escaped asterisk inside
    r"|(\*(?:\\\*|[^*])+\*)"            # italic, same
    r"|(\[[^\]]+\]\([^)]+\))"           # link
)


def _unesc(s):
    return _ESCAPED.sub(r"\1", s)


def inline(text):
    """Markdown inline -> HTML. Escapes HTML first so report content cannot inject markup."""
    out, last = [], 0
    for m in _INLINE.finditer(text):
        out.append(_unesc(html.escape(text[last:m.start()])))
        tok = m.group(0)
        if tok.startswith("`"):
            out.append("<code>%s</code>" % html.escape(tok[1:-1]))       # raw: no unescaping
        elif tok.startswith("**"):
            out.append("<strong>%s</strong>" % _unesc(html.escape(tok[2:-2])))
        elif tok.startswith("["):
            label, href = re.match(r"^\[([^\]]+)\]\(([^)]+)\)$", tok).groups()
            out.append('<a href="%s">%s</a>' % (html.escape(href, quote=True),
                                                _unesc(html.escape(label))))
        else:
            out.append("<em>%s</em>" % _unesc(html.escape(tok[1:-1])))
        last = m.end()
    out.append(_unesc(html.escape(text[last:])))
    return "".join(out)


def split_cells(line):
    """Split a table row on pipes that are not backslash-escaped."""
    cells, cur, i = [], "", 0
    while i < len(line):
        if line[i] == "\\" and i + 1 < len(line) and line[i + 1] == "|":
            cur += "|"
            i += 2
            continue
        if line[i] == "|":
            cells.append(cur)
            cur = ""
            i += 1
            continue
        cur += line[i]
        i += 1
    cells.append(cur)
    cells = [c.strip() for c in cells]
    if cells and not cells[0]:
        cells.pop(0)
    if cells and not cells[-1]:
        cells.pop()
    return cells


CSS = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm; }
* { box-sizing: border-box; }
body { font-family: "Calibri","Segoe UI",Arial,sans-serif; font-size: 10.5pt; line-height: 1.5;
       color: #1a1a1a; margin: 0; }
h1 { font-size: 26pt; color: #1F3864; margin: 0 0 6pt; font-weight: 700; letter-spacing: -.3pt; }
h2 { font-size: 15pt; color: #1F3864; margin: 20pt 0 7pt; font-weight: 700;
     border-bottom: 1.2pt solid #d4dae6; padding-bottom: 3pt;
     break-after: avoid; page-break-after: avoid; }
h3 { font-size: 12pt; color: #2E5496; margin: 14pt 0 5pt; font-weight: 700;
     break-after: avoid; page-break-after: avoid; }
h4 { font-size: 11pt; color: #2E5496; margin: 11pt 0 4pt; font-weight: 700; }
p { margin: 0 0 7pt; text-align: justify; hyphens: auto; }
ul, ol { margin: 0 0 8pt; padding-left: 18pt; }
li { margin-bottom: 3pt; }
code { font-family: Consolas,"Courier New",monospace; font-size: 9pt;
       background: #f2f4f7; padding: .5pt 2.5pt; border-radius: 2pt; }
pre { font-family: Consolas,"Courier New",monospace; font-size: 8.5pt; line-height: 1.35;
      background: #f5f7fa; border-left: 2.5pt solid #c3ccdb; padding: 7pt 9pt;
      margin: 0 0 9pt; white-space: pre-wrap; overflow-wrap: anywhere;
      break-inside: avoid; page-break-inside: avoid; }
table { border-collapse: collapse; width: 100%; margin: 0 0 10pt; font-size: 8.8pt;
        break-inside: avoid; page-break-inside: avoid; }
th { background: #e8eef4; text-align: left; font-weight: 700; }
th, td { border: .6pt solid #b9c3d1; padding: 3pt 4.5pt; vertical-align: top; }
tr:nth-child(even) td { background: #fafbfd; }
hr { border: 0; border-top: .8pt solid #cfd6e0; margin: 13pt 0; }
a { color: #1F3864; text-decoration: none; border-bottom: .4pt dotted #8b9ab0; }
strong { color: #10203d; }
"""


def convert(md_text):
    lines = md_text.split("\n")
    out, i = [], 0
    while i < len(lines):
        line = lines[i]

        if line.startswith("```"):                                  # fenced code
            i += 1
            block = []
            while i < len(lines) and not lines[i].startswith("```"):
                block.append(lines[i])
                i += 1
            i += 1
            out.append("<pre>%s</pre>" % html.escape("\n".join(block)))
            continue

        if line.startswith("|") and i + 1 < len(lines) \
                and re.match(r"^\|[\s:|-]+\|?\s*$", lines[i + 1]):  # table
            head = split_cells(line)
            i += 2
            body = []
            while i < len(lines) and lines[i].startswith("|"):
                body.append(split_cells(lines[i]))
                i += 1
            t = ["<table><thead><tr>"]
            t += ["<th>%s</th>" % inline(c) for c in head]
            t.append("</tr></thead><tbody>")
            for row in body:
                row = (row + [""] * len(head))[:len(head)]
                t.append("<tr>" + "".join("<td>%s</td>" % inline(c) for c in row) + "</tr>")
            t.append("</tbody></table>")
            out.append("".join(t))
            continue

        if re.match(r"^---+\s*$", line):                             # rule
            out.append("<hr>")
            i += 1
            continue

        m = re.match(r"^(#{1,4})\s+(.*)$", line)                     # heading
        if m:
            lvl = len(m.group(1))
            out.append("<h%d>%s</h%d>" % (lvl, inline(m.group(2)), lvl))
            i += 1
            continue

        if re.match(r"^\s*[-*]\s+", line):                           # bullet list
            items = []
            while i < len(lines) and re.match(r"^\s*[-*]\s+", lines[i]):
                txt = re.sub(r"^\s*[-*]\s+", "", lines[i])
                i += 1
                while i < len(lines) and re.match(r"^\s{2,}\S", lines[i]) \
                        and not re.match(r"^\s*[-*]\s+", lines[i]):
                    txt += " " + lines[i].strip()
                    i += 1
                items.append(txt)
            out.append("<ul>%s</ul>" % "".join("<li>%s</li>" % inline(t) for t in items))
            continue

        if re.match(r"^\s*\d+\.\s+", line):                          # numbered list
            items = []
            while i < len(lines) and re.match(r"^\s*\d+\.\s+", lines[i]):
                txt = re.sub(r"^\s*\d+\.\s+", "", lines[i])
                i += 1
                while i < len(lines) and re.match(r"^\s{2,}\S", lines[i]) \
                        and not re.match(r"^\s*\d+\.\s+", lines[i]):
                    txt += " " + lines[i].strip()
                    i += 1
                items.append(txt)
            out.append("<ol>%s</ol>" % "".join("<li>%s</li>" % inline(t) for t in items))
            continue

        if not line.strip():
            i += 1
            continue

        # paragraph. Only genuine BLOCK starts end one - a line may legitimately open with a
        # backtick (section 3.1 begins with `components()`), and treating that as a fence would
        # consume nothing and loop forever.
        def block_start(s):
            return (s.startswith("```") or re.match(r"^#{1,4}\s", s) or s.startswith("|")
                    or re.match(r"^---+\s*$", s) or re.match(r"^\s*[-*]\s+", s)
                    or re.match(r"^\s*\d+\.\s+", s))

        buf = []
        while i < len(lines) and lines[i].strip() and not block_start(lines[i]):
            buf.append(lines[i].strip())
            i += 1
        if buf:
            out.append("<p>%s</p>" % inline(" ".join(buf)))
        else:                                                        # guarantee progress
            out.append("<p>%s</p>" % inline(lines[i].strip()))
            i += 1

    title = "Report"
    for l in lines:
        if l.startswith("# "):
            title = l[2:].strip()
            break
    return ("<!doctype html><html><head><meta charset=\"utf-8\">"
            "<title>%s</title><style>%s</style></head><body>\n%s\n</body></html>"
            % (html.escape(title), CSS, "\n".join(out)))


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    with open(src, encoding="utf-8") as fh:
        page = convert(fh.read())
    with open(dst, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(page)
    print(f"wrote {dst} ({len(page):,} bytes)")
