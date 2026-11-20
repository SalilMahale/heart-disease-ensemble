"""Render paper_content.json to IEEE conference format (IEEEtran, two columns, A4).
Wide figures and all tables span both columns so nothing is squeezed."""
import json, os, re, sys, shutil
sys.path.insert(0, os.path.dirname(__file__))
from render_tex import esc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper_ieee")
os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
WIDE = 0.70          # figures at or above this fraction span both columns


def fig_path(p):
    dst = os.path.join(OUT, "figures", os.path.basename(p))
    if os.path.exists(p):
        shutil.copy(p, dst)
    return "figures/" + os.path.basename(p)


def render(b):
    t = b["type"]
    if t == "h1": return "\\section{%s}\n" % esc(re.sub(r"^\d+\s*", "", b["text"]))
    if t == "h2": return "\\subsection{%s}\n" % esc(re.sub(r"^\d+\.\d+\s*", "", b["text"]))
    if t == "h3": return "\\subsubsection{%s}\n" % esc(re.sub(r"^[\d.]+\s*", "", b["text"]))
    if t == "p": return esc(b["text"]) + "\n\n"
    if t == "bullets": return "\\begin{itemize}\n" + "".join("\\item %s\n" % esc(i) for i in b["items"]) + "\\end{itemize}\n"
    if t == "numbered": return "\\begin{enumerate}\n" + "".join("\\item %s\n" % esc(i) for i in b["items"]) + "\\end{enumerate}\n"
    if t == "eq": return "\\begin{equation}\n%s\n\\end{equation}\n" % b["latex"]
    if t == "pagebreak": return ""
    if t == "figure":
        star = "*" if b["width"] >= WIDE else ""
        w = "0.86\\textwidth" if star else "\\columnwidth"
        return ("\\begin{figure%s}[!t]\n\\centering\n\\includegraphics[width=%s]{%s}\n\\caption{%s}\n\\end{figure%s}\n\n"
                % (star, w, fig_path(b["path"]), esc(b["caption"]), star))
    if t == "table":
        n = len(b["header"]); w = b.get("widths") or [1] * n; tot = sum(w)
        cols = "".join(">{\\raggedright\\arraybackslash}p{%.3f\\textwidth}" % (0.90 * x / tot) for x in w)
        s = ("\\begin{table*}[!t]\n\\centering\n\\caption{%s}\n\\footnotesize\n\\begin{tabular}{%s}\n\\hline\n"
             % (esc(b["caption"]).upper(), cols))
        s += " & ".join("\\textbf{%s}" % esc(h) for h in b["header"]) + " \\\\\n\\hline\n"
        for r in b["rows"]:
            cells = [esc(str(c)) for c in r]
            if cells and cells[0].startswith("["):
                cells[0] = "{}" + cells[0]
            s += " & ".join(cells) + " \\\\\n"
        return s + "\\hline\n\\end{tabular}\n\\end{table*}\n\n"
    if t == "references":
        s = "\\begin{thebibliography}{99}\n"
        for i, r in enumerate(b["items"], 1):
            s += "\\bibitem{r%d} %s\n" % (i, esc(r))
        return s + "\\end{thebibliography}\n"
    raise ValueError(t)


def main():
    d = json.load(open(os.path.join(ROOT, "docs", "paper_content.json")))
    m = d["meta"]
    body = "".join(render(b) for b in d["blocks"])
    body = re.sub(r"\[(\d+(?:,\s*\d+)*)\]",
                  lambda mm: "\\cite{" + ",".join("r" + x.strip() for x in mm.group(1).split(",")) + "}", body)
    affil_lines = "\\\\".join(esc(x.strip()) for x in m["affil"].split(", India")[0].split(", ")) + "\\\\" + esc("Mumbai, India")
    tex = r"""\documentclass[conference,a4paper,10pt]{IEEEtran}
\IEEEoverridecommandlockouts
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{graphicx,amsmath,amssymb,array,cite,url}
\usepackage[hidelinks]{hyperref}
\begin{document}
\title{__TITLE__}
\author{\IEEEauthorblockN{__AUTHOR__}
\IEEEauthorblockA{__AFFIL__\\
\texttt{__EMAIL__}}}
\maketitle
\begin{abstract}
__ABSTRACT__
\end{abstract}
\begin{IEEEkeywords}
__KEYWORDS__
\end{IEEEkeywords}
__BODY__
\end{document}
"""
    for k, v in {"TITLE": m["title"], "AUTHOR": m["author"], "EMAIL": m["email"]}.items():
        tex = tex.replace("__%s__" % k, esc(v))
    tex = (tex.replace("__AFFIL__", affil_lines)
              .replace("__KEYWORDS__", esc(m["keywords"]))
              .replace("__ABSTRACT__", esc(d["abstract"]))
              .replace("__BODY__", body))
    open(os.path.join(OUT, "ML_MiniProject_Paper_IEEE.tex"), "w").write(tex)
    print("wrote IEEE tex")


if __name__ == "__main__":
    main()
