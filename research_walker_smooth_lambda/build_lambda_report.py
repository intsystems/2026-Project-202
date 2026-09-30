from pathlib import Path
import re,subprocess
H=Path(__file__).resolve().parent
def inline(s):
    chunks=re.split(r'(\$[^$]*\$|\*\*.*?\*\*)',s);out=[]
    for c in chunks:
        if c.startswith('$') and c.endswith('$'):out.append(c)
        elif c.startswith('**') and c.endswith('**'):out.append(r'\textbf{'+inline(c[2:-2])+'}')
        else:
            esc={'&':r'\&','%':r'\%','_':r'\_','#':r'\#','×':r'\ensuremath{\times}','–':'--','—':'---','→':r'\ensuremath{\to}','−':r'\ensuremath{-}'}
            out.append(''.join(esc.get(ch,ch) for ch in c))
    return ''.join(out)
lines=(H/'report_lambda.md').read_text(encoding='utf-8').splitlines();body=[];i=1
while i<len(lines):
    s=lines[i]
    if s=='<!-- pagebreak -->':body.append(r'\newpage')
    elif s=='$$':
        f=[];i+=1
        while lines[i]!='$$':f.append(lines[i]);i+=1
        body += [r'\[',*f,r'\]']
    elif s.startswith('## '):body.append(r'\section*{'+inline(s[3:])+r'}')
    elif s.startswith('|'):
        rows=[]
        while i<len(lines) and lines[i].startswith('|'):
            cells=[c.strip() for c in lines[i].strip('|').split('|')]
            if not all(re.fullmatch(r':?-+:?',c) for c in cells):rows.append(cells)
            i+=1
        i-=1;n=len(rows[0]);cols=r'l'+r'>{\raggedright\arraybackslash}X'*(n-1)
        body += [r'\begin{center}\small',r'\begin{tabularx}{\linewidth}{'+cols+'}',r'\toprule']
        for j,row in enumerate(rows):
            body.append(' & '.join(inline(c) for c in row)+r' \\')
            if j==0:body.append(r'\midrule')
        body += [r'\bottomrule\end{tabularx}\end{center}']
    elif s.startswith('!['):
        match=re.match(r'!\[(.*?)\]\((.*?)\)',s)
        body.append(r'\begin{center}\includegraphics[width=\linewidth,height=.32\textheight,keepaspectratio]{'+match[2]+r'}\end{center}')
    else:body.append(inline(s))
    i+=1
pre=r'''\documentclass[10pt,a4paper]{article}
\usepackage{fontspec,polyglossia}\setmainlanguage{russian}\setmainfont{Times New Roman}
\usepackage{amsmath,amssymb,booktabs,tabularx,graphicx}\usepackage[margin=17mm]{geometry}
\usepackage[hidelinks]{hyperref}\usepackage{titlesec}\titleformat{\section}{\large\bfseries}{}{0pt}{}
\titlespacing*{\section}{0pt}{8pt}{4pt}\setlength{\parindent}{0pt}\setlength{\parskip}{4pt}\setlength{\emergencystretch}{3em}
\begin{document}
'''
tex=pre+r'{\Large\bfseries '+inline(lines[0][2:])+r'}\par\medskip'+'\n'+'\n'.join(body)+'\n'+r'\end{document}'
(H/'report_lambda.tex').write_text(tex,encoding='utf-8')
for n in (1,2):
    r=subprocess.run(['xelatex','-interaction=nonstopmode','-halt-on-error','-file-line-error','report_lambda.tex'],cwd=H,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (H/f'build_lambda_{n}.log').write_bytes(r.stdout)
    if r.returncode:raise RuntimeError(f'XeLaTeX failed; see build_lambda_{n}.log')
log=(H/'report_lambda.log').read_text(encoding='utf-8',errors='replace')
issues=[l for l in log.splitlines() if any(t in l for t in ['Overfull','Missing character','LaTeX Error','Undefined control'])]
if issues:raise RuntimeError('\n'.join(issues))
print('Compiled with no missing glyphs, errors or overflows.')
