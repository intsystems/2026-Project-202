from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parent


def inline(text):
    chunks = re.split(r'(\$[^$]+\$|\\\([^\n]*?\\\)|`[^`]+`|\*\*.*?\*\*)', text)
    def convert(s):
        if s.startswith('$') and s.endswith('$'):
            return s
        if s.startswith(r'\(') and s.endswith(r'\)'):
            return '$' + s[2:-2] + '$'
        if s.startswith('`') and s.endswith('`'):
            return r'\nolinkurl{' + s[1:-1] + '}'
        if s.startswith('**') and s.endswith('**'):
            return r'\textbf{' + inline(s[2:-2]) + '}'
        table = {'&': r'\&', '%': r'\%', '_': r'\_', '#': r'\#',
                 '→': r'\ensuremath{\to}', '—': '---'}
        return ''.join(table.get(c, c) for c in s)
    return ''.join(convert(s) for s in chunks)


def main():
    lines = (ROOT / 'same_seed_spectrum_report_ru.md').read_text(encoding='utf-8').splitlines()
    title = lines[0][2:]
    body, i = [], 1
    while i < len(lines):
        line = lines[i]
        if line == '$$':
            eq = []
            i += 1
            while i < len(lines) and lines[i] != '$$':
                eq.append(lines[i]); i += 1
            body.extend([r'\[', *eq, r'\]'])
        elif line.startswith('## '):
            body.append(r'\section*{' + inline(line[3:]) + '}')
        elif line.startswith('|'):
            rows = []
            while i < len(lines) and lines[i].startswith('|'):
                cells = [c.strip() for c in lines[i].strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?', c) for c in cells):
                    rows.append(cells)
                i += 1
            i -= 1
            cols = len(rows[0])
            body += [r'\begin{center}\small',
                     r'\begin{tabularx}{\linewidth}{' +
                     r'>{\raggedright\arraybackslash}X' * cols + '}', r'\toprule']
            for j, row in enumerate(rows):
                body.append(' & '.join(inline(c) for c in row) + r' \\')
                if j == 0: body.append(r'\midrule')
            body += [r'\bottomrule\end{tabularx}\end{center}']
        elif line.startswith('!['):
            body += [r'\begin{center}',
                     r'\includegraphics[width=\linewidth]{same_seed_spectrum_MG_sweep.pdf}',
                     r'\end{center}']
        else:
            body.append(inline(line))
        i += 1
    pre = r'''\documentclass[10pt,a4paper]{article}
\usepackage{fontspec,polyglossia}
\setmainlanguage{russian}
\setmainfont{Times New Roman}
\usepackage{amsmath,amssymb,booktabs,tabularx,graphicx}
\usepackage[margin=19mm]{geometry}
\usepackage[hidelinks]{hyperref}
\setlength{\parindent}{0pt}
\setlength{\parskip}{4pt}
\setlength{\emergencystretch}{3em}
\author{}\date{}
'''
    tex = pre + r'\title{' + inline(title) + r'}' + '\n' + r'\begin{document}\maketitle' + '\n'
    tex += '\n'.join(body) + '\n' + r'\end{document}' + '\n'
    (ROOT / 'same_seed_spectrum_report_ru.tex').write_text(tex, encoding='utf-8')
    for run in (1, 2):
        result = subprocess.run(['xelatex', '-interaction=nonstopmode',
                                 '-halt-on-error', '-file-line-error',
                                 'same_seed_spectrum_report_ru.tex'],
                                cwd=ROOT, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        (ROOT / f'same_seed_report_build_{run}.log').write_bytes(result.stdout)
        if result.returncode:
            raise RuntimeError(f'XeLaTeX failed: same_seed_report_build_{run}.log')
    log = (ROOT / 'same_seed_spectrum_report_ru.log').read_text(encoding='utf-8', errors='replace')
    issues = [line for line in log.splitlines() if any(s in line for s in
              ('Overfull', 'Missing character', 'LaTeX Error', 'Undefined control'))]
    if issues:
        raise RuntimeError('\n'.join(issues))
    print('Built same_seed_spectrum_report_ru.pdf without LaTeX errors or overflow.')


if __name__ == '__main__':
    main()
