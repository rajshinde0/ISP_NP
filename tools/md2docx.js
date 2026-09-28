// Markdown -> .docx for the ISP Network Planner report.
// Handles: h1-h3, paragraphs with **bold**/*italic*/`code`/[links], tables (with \| escapes),
// fenced code blocks, bullet and numbered lists, and --- rules.
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell,
  WidthType, BorderStyle, AlignmentType, ShadingType, LevelFormat, PageOrientation,
  ExternalHyperlink,
} = require("docx");

const SRC = process.argv[2];
const OUT = process.argv[3];
const md = fs.readFileSync(SRC, "utf8").split(/\r?\n/);

// A4 (default) minus 0.8" margins each side, in DXA (1440 = 1 inch)
const MARGIN = 1152;
const CONTENT_W = 11906 - 2 * MARGIN;

// ---------- inline formatting ----------
// Strip markdown backslash-escapes (\*, \$, \|, \_ ...). Applied to prose, headings and table
// cells but NOT to code spans or fenced blocks, where a backslash is literal - the Appendix's
// `venv\Scripts\activate` must survive intact.
const unesc = (s) => s.replace(/\\([\\`*_{}\[\]()#+\-.!$|])/g, "$1");

function inlineRuns(text, base = {}) {
  const runs = [];
  // code first so ** inside `code` is not treated as bold
  // bold/italic bodies allow an ESCAPED asterisk, so "**A\***" is one bold token rather than
  // "**A\**" plus a stray "*" - that mis-parse is what leaked a literal backslash into "A*".
  const re = /(`[^`]+`)|(\*\*(?:\\\*|[^*])+\*\*)|(\*(?:\\\*|[^*])+\*)|(\[[^\]]+\]\([^)]+\))/g;
  let last = 0, m;
  const push = (t, opts) => { if (t) runs.push(new TextRun({ text: unesc(t), ...base, ...opts })); };
  const pushRaw = (t, opts) => { if (t) runs.push(new TextRun({ text: t, ...base, ...opts })); };
  while ((m = re.exec(text)) !== null) {
    push(text.slice(last, m.index), {});
    const tok = m[0];
    if (tok.startsWith("`")) {
      // raw: inside a code span a backslash means itself
      pushRaw(tok.slice(1, -1), { font: "Consolas", size: base.size ? base.size - 2 : 18 });
    } else if (tok.startsWith("**")) {
      push(tok.slice(2, -2), { bold: true });
    } else if (tok.startsWith("[")) {
      const mm = /^\[([^\]]+)\]\(([^)]+)\)$/.exec(tok);
      runs.push(new ExternalHyperlink({
        children: [new TextRun({ text: unesc(mm[1]), style: "Hyperlink", ...base })],
        link: mm[2],
      }));
    } else {
      push(tok.slice(1, -1), { italics: true });
    }
    last = m.index + tok.length;
  }
  push(text.slice(last), {});
  return runs.length ? runs : [new TextRun({ text: "", ...base })];
}

// split a table row on pipes that are not backslash-escaped
function splitCells(line) {
  const out = [];
  let cur = "";
  for (let i = 0; i < line.length; i++) {
    if (line[i] === "\\" && line[i + 1] === "|") { cur += "|"; i++; continue; }
    if (line[i] === "|") { out.push(cur); cur = ""; continue; }
    cur += line[i];
  }
  out.push(cur);
  return out.map(s => s.trim()).filter((_, i, a) => !(i === 0 && a[0] === "") && !(i === a.length - 1 && a[a.length - 1] === ""));
}

function buildTable(rows) {
  const header = splitCells(rows[0]);
  const body = rows.slice(2).map(splitCells);
  const n = header.length;
  const colW = Math.floor(CONTENT_W / n);
  const widths = Array(n).fill(colW);
  widths[n - 1] = CONTENT_W - colW * (n - 1);

  const cell = (text, isHeader) => new TableCell({
    width: { size: widths[0], type: WidthType.DXA },
    shading: isHeader ? { type: ShadingType.CLEAR, fill: "E8EEF4" } : undefined,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({
      spacing: { before: 20, after: 20 },
      children: inlineRuns(text, { size: 17, bold: isHeader || undefined }),
    })],
  });

  const mkRow = (cells, isHeader) => new TableRow({
    tableHeader: isHeader || undefined,
    children: cells.concat(Array(Math.max(0, n - cells.length)).fill("")).slice(0, n)
      .map((t, i) => new TableCell({
        width: { size: widths[i], type: WidthType.DXA },
        shading: isHeader ? { type: ShadingType.CLEAR, fill: "E8EEF4" } : undefined,
        margins: { top: 60, bottom: 60, left: 100, right: 100 },
        children: [new Paragraph({
          spacing: { before: 20, after: 20 },
          children: inlineRuns(t, { size: 17, bold: isHeader || undefined }),
        })],
      })),
  });

  return new Table({
    columnWidths: widths,
    width: { size: CONTENT_W, type: WidthType.DXA },
    rows: [mkRow(header, true)].concat(body.map(r => mkRow(r, false))),
  });
}

// ---------- block parsing ----------
const children = [];
let i = 0;
while (i < md.length) {
  const line = md[i];

  // fenced code block
  if (/^```/.test(line)) {
    i++;
    const code = [];
    while (i < md.length && !/^```/.test(md[i])) code.push(md[i++]);
    i++;
    code.forEach((c, idx) => children.push(new Paragraph({
      spacing: { before: idx === 0 ? 120 : 0, after: idx === code.length - 1 ? 120 : 0 },
      shading: { type: ShadingType.CLEAR, fill: "F4F6F8" },
      children: [new TextRun({ text: c || " ", font: "Consolas", size: 16 })],
    })));
    continue;
  }

  // table
  if (/^\|/.test(line) && i + 1 < md.length && /^\|[\s:|-]+\|?\s*$/.test(md[i + 1])) {
    const rows = [];
    while (i < md.length && /^\|/.test(md[i])) rows.push(md[i++]);
    children.push(buildTable(rows));
    children.push(new Paragraph({ spacing: { after: 160 }, children: [] }));
    continue;
  }

  // horizontal rule
  if (/^---+\s*$/.test(line)) {
    children.push(new Paragraph({
      spacing: { before: 120, after: 120 },
      border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "BBBBBB" } },
      children: [],
    }));
    i++; continue;
  }

  // headings
  let h = /^(#{1,4})\s+(.*)$/.exec(line);
  if (h) {
    const lvl = [HeadingLevel.TITLE, HeadingLevel.HEADING_1, HeadingLevel.HEADING_2,
                 HeadingLevel.HEADING_3][h[1].length - 1];
    children.push(new Paragraph({
      heading: lvl,
      spacing: { before: h[1].length === 1 ? 0 : 240, after: 120 },
      children: inlineRuns(h[2]),
    }));
    i++; continue;
  }

  // bullet list
  if (/^\s*[-*]\s+/.test(line)) {
    while (i < md.length && /^\s*[-*]\s+/.test(md[i])) {
      const indent = /^(\s*)/.exec(md[i])[1].length >= 2 ? 1 : 0;
      children.push(new Paragraph({
        bullet: { level: indent },
        spacing: { after: 60 },
        children: inlineRuns(md[i].replace(/^\s*[-*]\s+/, "")),
      }));
      i++;
      // continuation lines of the same bullet
      while (i < md.length && /^\s{2,}\S/.test(md[i]) && !/^\s*[-*]\s+/.test(md[i])) {
        children.push(new Paragraph({
          bullet: { level: indent }, spacing: { after: 60 },
          children: inlineRuns(md[i].trim()),
        }));
        i++;
      }
    }
    continue;
  }

  // numbered list
  if (/^\s*\d+\.\s+/.test(line)) {
    while (i < md.length && (/^\s*\d+\.\s+/.test(md[i]) || /^\s{2,}\S/.test(md[i]))) {
      const txt = md[i].replace(/^\s*\d+\.\s+/, "").trim();
      children.push(new Paragraph({
        numbering: { reference: "num", level: 0 },
        spacing: { after: 60 },
        children: inlineRuns(txt),
      }));
      i++;
    }
    continue;
  }

  // blank
  if (!line.trim()) { i++; continue; }

  // paragraph (join wrapped lines).
  // Only genuine BLOCK starts may end a paragraph. A line beginning with a backtick is almost
  // always inline code (`components()` opens a paragraph in section 3.1), not a fence - testing
  // /^[#|`]/ here let such a line match nothing and never advance i, which hung the converter.
  const isBlockStart = (s) =>
    /^```/.test(s) || /^#{1,4}\s/.test(s) || /^\|/.test(s) || /^---+\s*$/.test(s) ||
    /^\s*[-*]\s+/.test(s) || /^\s*\d+\.\s+/.test(s);
  const buf = [];
  while (i < md.length && md[i].trim() && !isBlockStart(md[i])) {
    buf.push(md[i].trim()); i++;
  }
  if (buf.length) {
    children.push(new Paragraph({
      spacing: { after: 120, line: 280 },
      alignment: AlignmentType.JUSTIFIED,
      children: inlineRuns(buf.join(" ")),
    }));
  } else {
    // Belt and braces: nothing consumed this pass, so take one line and move on. Without this
    // any future unhandled block start is an infinite loop rather than a cosmetic glitch.
    children.push(new Paragraph({
      spacing: { after: 120 }, children: inlineRuns(md[i].trim()),
    }));
    i++;
  }
}

const doc = new Document({
  numbering: {
    config: [{
      reference: "num",
      levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.START,
                 style: { paragraph: { indent: { left: 720, hanging: 360 } } } }],
    }],
  },
  styles: {
    default: { document: { run: { font: "Calibri", size: 21 } } },
    paragraphStyles: [
      { id: "Title", name: "Title", basedOn: "Normal", next: "Normal",
        run: { size: 40, bold: true, color: "1F3864" }, paragraph: { spacing: { after: 200 } } },
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal",
        run: { size: 30, bold: true, color: "1F3864" }, paragraph: { spacing: { before: 280, after: 140 } } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal",
        run: { size: 25, bold: true, color: "2E5496" }, paragraph: { spacing: { before: 240, after: 120 } } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal",
        run: { size: 22, bold: true, color: "2E5496" }, paragraph: { spacing: { before: 200, after: 100 } } },
    ],
  },
  sections: [{
    properties: { page: { margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN } } },
    children,
  }],
});

Packer.toBuffer(doc).then(b => {
  fs.writeFileSync(OUT, b);
  console.log(`wrote ${OUT} (${b.length.toLocaleString()} bytes, ${children.length} blocks)`);
});
