"""Render report_content.json to LaTeX (report style, Times New Roman look)."""
import json, os, re, sys, shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "report_latex")
os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)


def esc(s):
    """Escape LaTeX specials, then turn **bold** into \\textbf{}."""
    s = s.replace("\\", r"\textbackslash{}")
    for a, b in [("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"),
                 ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}"),
                 ("\u00b1", r"$\pm$"), ("<", r"\textless{}"), (">", r"\textgreater{}"), ("|", r"\textbar{}")]:
        s = s.replace(a, b)
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
    return s


def fig_path(p):
    dst = os.path.join(OUT, "figures", os.path.basename(p))
    if os.path.exists(p):
        shutil.copy(p, dst)
    return "figures/" + os.path.basename(p)


def render_block(b):
    t = b["type"]
    if t == "h1":
        title = re.sub(r"^Chapter \d+:\s*", "", b["text"])
        return "\\chapter{%s}\n" % esc(title)
    if t == "h2":
        return "\\section{%s}\n" % esc(re.sub(r"^\d+\.\d+\s*", "", b["text"]))
    if t == "h3":
        return "\\subsection{%s}\n" % esc(re.sub(r"^\d+\.\d+\.\d+\s*", "", b["text"]))
    if t == "p":
        return esc(b["text"]) + "\n\n"
    if t == "bullets":
        return "\\begin{itemize}\n" + "".join("  \\item %s\n" % esc(i) for i in b["items"]) + "\\end{itemize}\n\n"
    if t == "numbered":
        return "\\begin{enumerate}\n" + "".join("  \\item %s\n" % esc(i) for i in b["items"]) + "\\end{enumerate}\n\n"
    if t == "eq":
        return "\\begin{equation}\n%s\n\\end{equation}\n\n" % b["latex"]
    if t == "pagebreak":
        return ""   # chapters start on new pages already
    if t == "figure":
        return ("\\begin{figure}[htbp]\n\\centering\n\\includegraphics[width=%.2f\\textwidth]{%s}\n\\caption{%s}\n\\end{figure}\n\n"
                % (min(b["width"], 1.0), fig_path(b["path"]), esc(b["caption"])))
    if t == "table":
        n = len(b["header"])
        widths = b.get("widths") or [1] * n
        tot = sum(widths)
        cols = "".join(">{\\raggedright\\arraybackslash}p{%.3f\\textwidth}" % (0.94 * w / tot) for w in widths)
        s = "\\begin{table}[htbp]\n\\centering\n\\caption{%s}\n\\footnotesize\n\\begin{tabular}{%s}\n\\hline\n" % (esc(b["caption"]), cols)
        s += " & ".join("\\textbf{%s}" % esc(h) for h in b["header"]) + " \\\\\n\\hline\n"
        for r in b["rows"]:
            cells = [esc(str(c)) for c in r]
            if cells and cells[0].startswith("["):
                cells[0] = "{}" + cells[0]          # stop \\ reading [n] as an optional argument
            s += " & ".join(cells) + " \\\\\n"
        s += "\\hline\n\\end{tabular}\n\\end{table}\n\n"
        return s
    if t == "code":
        cap = ("caption={%s}," % esc(b["caption"])) if b.get("caption") else ""
        return "\\begin{lstlisting}[%s]\n%s\n\\end{lstlisting}\n\n" % (cap, b["text"])
    if t == "references":
        s = "\\begin{enumerate}[label={[\\arabic*]},leftmargin=2.2em]\n"
        for r in b["items"]:
            s += "  \\item %s\n" % esc(r)
        return s + "\\end{enumerate}\n"
    raise ValueError(t)


def main():
    d = json.load(open(os.path.join(ROOT, "docs", "report_content.json")))
    m = d["meta"]
    if m.get("logo") and os.path.exists(m["logo"]):
        shutil.copy(m["logo"], os.path.join(OUT, "figures", "sfit_logo.png"))
    body = "".join(render_block(b) for b in d["blocks"])
    abbrev = "\\begin{tabular}{ll}\n" + "".join("\\textbf{%s} & %s \\\\\n" % (esc(a), esc(f)) for a, f in d["abbrev"]) + "\\end{tabular}\n"
    tex = r"""\documentclass[12pt,a4paper,oneside]{report}
\usepackage[a4paper,margin=1in]{geometry}
\usepackage{mathptmx}            % Times New Roman text and maths
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{graphicx,float,array,booktabs,enumitem,amsmath,amssymb,xcolor}
\usepackage{caption}
\usepackage{listings}
\usepackage{setspace}
\setlength{\textfloatsep}{10pt plus 2pt minus 2pt}
\setlength{\intextsep}{10pt plus 2pt minus 2pt}
\setlength{\floatsep}{10pt plus 2pt minus 2pt}
\renewcommand{\topfraction}{0.9}\renewcommand{\bottomfraction}{0.8}
\renewcommand{\textfraction}{0.08}\renewcommand{\floatpagefraction}{0.75}
\usepackage{titlesec}
\usepackage[hidelinks]{hyperref}
\setstretch{1.0}
\setlength{\parskip}{4pt}
\setlength{\parindent}{0pt}
\captionsetup{font={small},labelfont={bf},justification=centering,skip=4pt}
\titleformat{\chapter}[hang]{\normalfont\fontsize{14}{16}\bfseries}{\chaptertitlename\ \thechapter:}{0.5em}{}
\titlespacing*{\chapter}{0pt}{0pt}{10pt}
\titleformat{\section}{\normalfont\fontsize{14}{16}\bfseries}{\thesection}{0.6em}{}
\titleformat{\subsection}{\normalfont\fontsize{12}{14}\bfseries}{\thesubsection}{0.6em}{}
\renewcommand{\listfigurename}{List of Figures}
\lstset{basicstyle=\ttfamily\scriptsize,breaklines=true,frame=single,columns=fullflexible,
  captionpos=b,showstringspaces=false,tabsize=4,language=Python,
  commentstyle=\color{black!60},keywordstyle=\bfseries,breakatwhitespace=false}
\pagestyle{plain}
\begin{document}
\pagenumbering{roman}
% ---------------- cover ----------------
\begin{titlepage}
\centering
{\fontsize{14}{16}\bfseries __DEPT__\par}
\vspace{4pt}
{\fontsize{14}{16}\bfseries __COLLEGE__\par}
\vspace{4pt}
{\fontsize{14}{16}\bfseries __UNIV__\par}
\vspace{4pt}
{\fontsize{14}{16}\bfseries __YEAR__\par}
\vspace{1.4cm}
{\fontsize{14}{16}\bfseries A Mini Project Report On\par}
\vspace{1cm}
{\fontsize{14}{16}\bfseries ``__TITLE__''\par}
\vspace{1.2cm}
{\fontsize{14}{16}\bfseries Subject: __SUBJECT__\par}
\vspace{1cm}
\includegraphics[width=3.4cm]{figures/sfit_logo.png}\par
\vspace{1cm}
{\fontsize{14}{16}\bfseries Under the guidance of\par}
\vspace{4pt}
{\fontsize{14}{16}\bfseries __GUIDE__\par}
\vspace{1cm}
{\fontsize{14}{16}\bfseries By\par}
\vspace{4pt}
{\fontsize{14}{16}\bfseries __STUDENT__ (Roll no. __ROLL__, PID no. __PID__)\par}
{\fontsize{14}{16}\bfseries __CLASS__, Batch __BATCH__\par}
\vfill
\end{titlepage}
\clearpage
\chapter*{Abstract}
\addcontentsline{toc}{chapter}{Abstract}
__ABSTRACT__
\vspace{1.5em}
\par\vspace{0.5em}
{\small\setlength{\parskip}{0pt}\let\clearpage\relax\par\tableofcontents}
\clearpage
\listoffigures
\addcontentsline{toc}{chapter}{List of Figures}
\par\vspace{1.2em}\par
{\let\clearpage\relax \listoftables}
\addcontentsline{toc}{chapter}{List of Tables}
\par\vspace{1.4em}
\noindent{\fontsize{14}{16}\bfseries List of Abbreviations}\par\vspace{0.6em}
\addcontentsline{toc}{chapter}{List of Abbreviations}
__ABBREV__
\clearpage
\pagenumbering{arabic}
__BODY__
\end{document}
"""
    for k, v in {"DEPT": m["dept"], "COLLEGE": m["college"], "UNIV": m["univ"], "YEAR": m["year"], "TITLE": m["title"],
                 "SUBJECT": m["subject"], "GUIDE": m["guide"], "STUDENT": m["student"], "ROLL": m["roll"], "PID": m["pid"],
                 "CLASS": m["class"], "BATCH": m["batch"]}.items():
        tex = tex.replace("__%s__" % k, esc(v))
    tex = tex.replace("__ABSTRACT__", esc(d["abstract"])).replace("__ABBREV__", abbrev).replace("__BODY__", body)
    open(os.path.join(OUT, "ML_MiniProject_Report.tex"), "w").write(tex)
    print("wrote", os.path.join(OUT, "ML_MiniProject_Report.tex"))


if __name__ == "__main__":
    main()
