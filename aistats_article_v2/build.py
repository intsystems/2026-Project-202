"""Build and verify both isolated AISTATS revisions from the same source."""
from pathlib import Path
import json,re,subprocess,shutil,xml.etree.ElementTree as ET
H=Path(__file__).resolve().parent
def run(cmd):
    p=subprocess.run(cmd,cwd=H,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    if p.returncode:raise RuntimeError(p.stdout.decode('utf-8',errors='replace')[-7000:])
    return p.stdout

results={}
for stem in ['aistats2027','aistats2027_blue']:
    out=run(['latexmk','-pdf','-interaction=nonstopmode','-halt-on-error','-file-line-error',stem+'.tex'])
    (H/(stem+'_build.log')).write_bytes(out)
    log=(H/(stem+'.log')).read_text(encoding='utf-8',errors='replace')
    bad=[s for s in log.splitlines() if any(x in s for x in ['Overfull','Missing character','undefined references','undefined citations','LaTeX Error'])]
    aux=(H/(stem+'.aux')).read_text(encoding='utf-8',errors='replace')
    m=re.search(r'\\newlabel\{LastMainPage\}\{\{[^}]*\}\{(\d+)\}',aux)
    if not m:raise RuntimeError('Main-text end label missing')
    page=int(m.group(1))
    results[stem]={'main_text_pages':page,'issues':bad}
    if page>8:raise RuntimeError(f'{stem}: main text ends on page {page}, limit is 8')
    if bad:raise RuntimeError('\n'.join(bad))
    run(['pdftotext','-layout',stem+'.pdf',stem+'_extracted.txt'])
black=(H/'aistats2027_extracted.txt').read_text(encoding='utf-8')
blue=(H/'aistats2027_blue_extracted.txt').read_text(encoding='utf-8')
assert black==blue,'Black/blue text differs'
# Inspect actual PDF graphics operators, not just the source macro.
try:
    import fitz
    counts={}
    for stem in results:
        doc=fitz.open(H/(stem+'.pdf'));spans=[s for page in doc for b in page.get_text('dict')['blocks'] if 'lines' in b for l in b['lines'] for s in l['spans']]
        blue_spans=sum(s['color']==0x195AA0 for s in spans)
        counts[stem]=blue_spans
        results[stem]['pdf_pages']=len(doc)
    assert counts['aistats2027_blue']>500 and counts['aistats2027']==0,counts
    results['verified_blue_spans']=counts
except ImportError:
    counts={}
    for stem in ['aistats2027','aistats2027_blue']:
        run(['pdftohtml','-xml','-hidden','-i',stem+'.pdf',stem+'_colors.xml'])
        root=ET.parse(H/(stem+'_colors.xml')).getroot()
        colors={f.attrib['id']:f.attrib.get('color','').lower() for f in root.iter('fontspec')}
        counts[stem]=sum(colors.get(t.attrib.get('font'))=='#195aa0' for t in root.iter('text'))
        results[stem]['pdf_pages']=len(root.findall('page'))
    assert counts['aistats2027_blue']>100 and counts['aistats2027']==0,counts
    results['verified_blue_text_blocks']=counts
(H/'document_text_equal.txt').write_text('Black and blue PDF text is identical.\n',encoding='utf-8')
(H/'build_validation.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
print(json.dumps(results,indent=2))
