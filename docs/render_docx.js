// Render report_content.json (or any blocks JSON) to .docx in the college format:
// Times New Roman, 12pt body/tables/code, 14pt headings, justified, no italics, black,
// no headers/footers, bold keywords, line spacing 1.
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow, TableCell,
  WidthType, BorderStyle, ImageRun, PageBreak, TableOfContents, LevelFormat, ShadingType, TabStopType, SectionType,
} = require("docx");

const [,, inJson, outDocx, mode = "report"] = process.argv;
const TWO_COL = mode === "paper2col";        // IEEE-style two-column paper
const PAPERISH = mode === "paper" || TWO_COL;
const d = JSON.parse(fs.readFileSync(inJson, "utf8"));
const FONT = "Times New Roman";
const SZ = TWO_COL ? 20 : 24, SZH = TWO_COL ? 24 : 28;   // half points; 10/12 pt in two-column, 12/14 pt otherwise
const PAGE_W = TWO_COL ? 4350 : 9026;   // usable width of one column vs the full text block
const FULL_W = 9026;             // A4 width minus 2 x 1" margins (DXA)

// ---------- inline text with **bold** ----------
function runs(text, extra = {}) {
  const out = [];
  const parts = String(text).split(/(\*\*.+?\*\*)/g);
  for (const p of parts) {
    if (!p) continue;
    const m = p.match(/^\*\*(.+)\*\*$/);
    out.push(new TextRun({ text: m ? m[1] : p, bold: !!m || !!extra.bold, font: FONT, size: extra.size || SZ, color: "000000" }));
  }
  return out;
}
function para(text, opts = {}) {
  return new Paragraph({
    alignment: opts.align || AlignmentType.JUSTIFIED,
    spacing: { after: opts.after ?? 120, line: 240 },
    keepNext: opts.keepNext,
    pageBreakBefore: opts.pageBreakBefore,
    children: runs(text, opts),
    numbering: opts.numbering,
    heading: opts.heading,
    outlineLevel: opts.outlineLevel,
  });
}
function heading(text, level) {
  const map = { 1: HeadingLevel.HEADING_1, 2: HeadingLevel.HEADING_2, 3: HeadingLevel.HEADING_3 };
  return new Paragraph({
    heading: map[level], alignment: AlignmentType.LEFT, keepNext: true,
    spacing: { before: level === 1 ? 0 : 200, after: 120, line: 240 },
    children: [new TextRun({ text, bold: true, font: FONT, size: level === 3 ? SZ : SZH, color: "000000" })],
  });
}
function cell(text, width, bold = false, shade = false) {
  return new TableCell({
    width: { size: width, type: WidthType.DXA },
    shading: shade ? { type: ShadingType.CLEAR, fill: "E7E6E6", color: "auto" } : undefined,
    margins: { top: 40, bottom: 40, left: 80, right: 80 },
    children: [new Paragraph({ alignment: AlignmentType.LEFT, spacing: { after: 0, line: 240 }, children: runs(text, { bold }) })],
  });
}
function table(header, rows, widths, full = false) {
  const TW = full ? FULL_W : PAGE_W;
  const n = header.length;
  const w = widths && widths.length === n ? widths : Array(n).fill(1);
  const tot = w.reduce((a, b) => a + b, 0);
  const cols = w.map(x => Math.round(TW * x / tot));
  cols[cols.length - 1] += TW - cols.reduce((a, b) => a + b, 0);
  const border = { style: BorderStyle.SINGLE, size: 4, color: "000000" };
  return new Table({
    width: { size: TW, type: WidthType.DXA }, columnWidths: cols,
    borders: { top: border, bottom: border, left: border, right: border, insideHorizontal: border, insideVertical: border },
    rows: [
      new TableRow({ tableHeader: true, children: header.map((h, i) => cell(h, cols[i], true, true)) }),
      ...rows.map(r => new TableRow({ children: r.map((c, i) => cell(String(c), cols[i])) })),
    ],
  });
}
function caption(text) {
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 60, after: 160, line: 240 },
    children: [new TextRun({ text, bold: true, font: FONT, size: SZ, color: "000000" })] });
}
function figure(p, widthFrac, full = false) {
  const buf = fs.readFileSync(p);
  const { width, height } = pngSize(buf);
  const base = (TWO_COL && !full) ? 3.0 : 6.2;   // inches available
  const wIn = Math.min(base * widthFrac, base);
  const hIn = wIn * height / width;
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 40 }, keepNext: true,
    children: [new ImageRun({ type: "png", data: buf, transformation: { width: Math.round(wIn * 96), height: Math.round(hIn * 96) } })] });
}
function pngSize(buf) { return { width: buf.readUInt32BE(16), height: buf.readUInt32BE(20) }; }
function codeBlock(text) {
  return text.split("\n").map(line => new Paragraph({
    alignment: AlignmentType.LEFT, spacing: { after: 0, line: 240 },
    shading: { type: ShadingType.CLEAR, fill: "F2F2F2", color: "auto" },
    children: [new TextRun({ text: line.replace(/\t/g, "    ") || " ", font: FONT, size: SZ, color: "000000" })],
  }));
}

// ---------- numbering (figures / tables by chapter) ----------
let chap = 0, figN = 0, tabN = 0;
var numRefs = [];
const figList = [], tabList = [];
const body = [];
const marks = [];                              // index in body -> "wide" when it must span the page
function pushWide(el) { marks[body.length] = "wide"; body.push(el); }
for (const b of d.blocks) {
  switch (b.type) {
    case "h1": { const m = b.text.match(/^Chapter (\d+)/); chap = m ? +m[1] : chap + 1; if (!PAPERISH) { figN = 0; tabN = 0; }
      body.push(heading(b.text, 1)); break; }
    case "h2": body.push(heading(b.text, 2)); break;
    case "h3": body.push(heading(b.text, 3)); break;
    case "p": body.push(para(b.text)); break;
    case "pagebreak": body.push(new Paragraph({ children: [new PageBreak()] })); break;
    case "bullets": for (const it of b.items) body.push(para(it, { numbering: { reference: "bul", level: 0 }, after: 60 })); break;
    case "numbered": { const ref = "num" + (numRefs.length); numRefs.push(ref);
      for (const it of b.items) body.push(para(it, { numbering: { reference: ref, level: 0 }, after: 60 })); break; }
    case "eq": body.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 60, after: 120, line: 240 },
      children: [new TextRun({ text: b.plain, font: FONT, size: SZ, color: "000000" })] })); break;
    case "figure": { figN++; const lab = PAPERISH ? `Fig. ${figN}. ${b.caption}` : `Figure ${chap}.${figN}: ${b.caption}`; figList.push(lab);
      const wide = TWO_COL && b.width >= 0.70;
      const f = figure(b.path, b.width, wide);
      if (wide) { pushWide(f); pushWide(caption(lab)); } else { body.push(f); body.push(caption(lab)); }
      break; }
    case "table": { tabN++; const lab = PAPERISH ? `Table ${tabN}. ${b.caption}` : `Table ${chap}.${tabN}: ${b.caption}`; tabList.push(lab);
      const cap = new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60, line: 240 }, keepNext: true,
        children: [new TextRun({ text: lab, bold: true, font: FONT, size: SZ, color: "000000" })] });
      const tb = table(b.header, b.rows, b.widths, TWO_COL);
      if (TWO_COL) { pushWide(cap); pushWide(tb); pushWide(new Paragraph({ spacing: { after: 120 } })); }
      else { body.push(cap); body.push(tb); body.push(new Paragraph({ spacing: { after: 120 } })); }
      break; }
    case "code": { if (b.caption) body.push(para(`Listing: ${b.caption}`, { bold: true, align: AlignmentType.LEFT, after: 40, keepNext: true }));
      body.push(...codeBlock(b.text)); body.push(new Paragraph({ spacing: { after: 120 } })); break; }
    case "references": b.items.forEach((r, i) => body.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED,
      spacing: { after: 100, line: 240 }, indent: { left: 560, hanging: 560 },
      children: [new TextRun({ text: `[${i + 1}] `, font: FONT, size: SZ, color: "000000" }), ...runs(r)] }))); break;
    default: throw new Error("unknown block " + b.type);
  }
}

function buildNumbering() {
  const cfg = [{ reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT,
    style: { paragraph: { indent: { left: 560, hanging: 280 } } } }] }];
  for (const r of numRefs) cfg.push({ reference: r, levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
    style: { paragraph: { indent: { left: 560, hanging: 360 } } } }] });
  return { config: cfg };
}

// ---------- front matter ----------
const m = d.meta;
const c = (t, opts = {}) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: opts.after ?? 120, line: 240 },
  children: [new TextRun({ text: t, bold: true, font: FONT, size: SZH, color: "000000" })] });
const gap = (n) => Array.from({ length: n }, () => new Paragraph({ spacing: { after: 0 }, children: [new TextRun({ text: " ", font: FONT, size: SZ })] }));
const cover = [
  c(m.dept), c(m.college), c(m.univ), c(m.year), ...gap(2),
  c("A Mini Project Report On"), ...gap(1), c(`\u201C${m.title}\u201D`), ...gap(1),
  c(`Subject: ${m.subject}`), ...gap(1),
  ...(m.logo && fs.existsSync(m.logo) ? [new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 120 },
      children: [new ImageRun({ type: "png", data: fs.readFileSync(m.logo), transformation: { width: 130, height: 127 } })] })] : []),
  ...gap(1), c("Under the guidance of"), c(m.guide), ...gap(2),
  c("By"), c(`${m.student} (Roll no. ${m.roll}, PID no. ${m.pid})`), c(`${m.class}, Batch ${m.batch}`),
  new Paragraph({ children: [new PageBreak()] }),
];
const front = [
  heading("Abstract", 1), para(d.abstract), new Paragraph({ children: [new PageBreak()] }),
  heading("Table of Contents", 1),
  new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-3" }),
  para("(Right click the table and choose Update Field to refresh page numbers after opening in Word.)", { align: AlignmentType.LEFT }),
  new Paragraph({ children: [new PageBreak()] }),
  heading("List of Figures", 1), ...figList.map(f => para(f, { align: AlignmentType.LEFT, after: 60 })),
  new Paragraph({ children: [new PageBreak()] }),
  heading("List of Tables", 1), ...tabList.map(f => para(f, { align: AlignmentType.LEFT, after: 60 })),
  new Paragraph({ spacing: { after: 200 } }),
  heading("List of Abbreviations", 1), table(["Abbreviation", "Full form"], d.abbrev, [2500, 6500]),
  new Paragraph({ children: [new PageBreak()] }),
];

const paperHead = !PAPERISH ? [] : [
  c(m.title), new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 }, children: [new TextRun({ text: m.author, font: FONT, size: SZ, bold: true, color: "000000" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 }, children: [new TextRun({ text: m.affil, font: FONT, size: SZ, color: "000000" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 }, children: [new TextRun({ text: m.email, font: FONT, size: SZ, color: "000000" })] }),
  para("**Abstract.** " + d.abstract), para("**Keywords:** " + m.keywords, { after: 240 }),
];
const PAGE = { margin: { top: 1440, bottom: 1080, left: 1080, right: 1080 } };
function sec(children, cols, first = false) {
  return { properties: { page: PAGE, column: cols > 1 ? { count: 2, space: 340 } : { count: 1 },
                         type: first ? undefined : SectionType.CONTINUOUS }, children };
}
function buildSections() {
  if (!TWO_COL) {
    return [{ properties: { page: { margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
              children: mode === "paper" ? [...paperHead, ...body] : [...cover, ...front, ...body] }];
  }
  const out = [sec(paperHead, 1, true)];            // title, authors, abstract across the page
  let buf = [], bufWide = marks[0] === "wide";
  for (let i = 0; i < body.length; i++) {
    const wide = marks[i] === "wide";
    if (wide !== bufWide) { if (buf.length) out.push(sec(buf, bufWide ? 1 : 2)); buf = []; bufWide = wide; }
    buf.push(body[i]);
  }
  if (buf.length) out.push(sec(buf, bufWide ? 1 : 2));
  return out;
}
const doc = new Document({
  creator: m.student, title: m.title,
  styles: {
    default: { document: { run: { font: FONT, size: SZ, color: "000000" } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: SZH, bold: true, font: FONT, color: "000000" }, paragraph: { spacing: { before: 0, after: 120 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: SZH, bold: true, font: FONT, color: "000000" }, paragraph: { spacing: { before: 200, after: 120 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: SZ, bold: true, font: FONT, color: "000000" }, paragraph: { spacing: { before: 160, after: 100 }, outlineLevel: 2 } },
    ],
  },
  numbering: buildNumbering(),
  features: { updateFields: true },
  sections: buildSections(),
});
Packer.toBuffer(doc).then(buf => { fs.writeFileSync(outDocx, buf); console.log("wrote", outDocx); });
