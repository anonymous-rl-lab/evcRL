"""Build v27 Markdown as IEEEtran main and one-column supplementary PDFs.
Requires Python 3, pandoc and pdflatex. Does not train or evaluate a policy.
"""
from pathlib import Path
import re, subprocess, functools, shutil, json
P=Path(__file__).resolve().parent.parent
L=P/'latex'

def clean(s):
    for a,b in {'−':'-','–':'--','—':'---','’':"'",'‘':"'",'“':'"','”':'"','ᵈ':r'\textsuperscript{d}','²':r'\textsuperscript{2}','³':r'\textsuperscript{3}','¹':r'\textsuperscript{1}','⁻':r'\textsuperscript{-}','±':r'\ensuremath{\pm}','×':r'\ensuremath{\times}','→':r'\ensuremath{\rightarrow}','≤':r'\ensuremath{\le}','≥':r'\ensuremath{\ge}','≈':r'\ensuremath{\approx}','°':r'\textdegree{}'}.items():s=s.replace(a,b)
    return s

def citations(s):return re.sub(r'\[((?:R-|S)[A-Za-z0-9-]+)\]',lambda m:r'\cite{'+m[1]+'}',s)
@functools.lru_cache(None)
def tex(s):
    if s.strip() in ['—','–','---','--']:return r'\textemdash{}'
    q=subprocess.run(['pandoc','-f','markdown+tex_math_dollars+raw_tex','-t','latex','--wrap=none'],input=clean(citations(s)),text=True,capture_output=True,check=True)
    t=q.stdout.strip()
    return re.sub(r'\\texttt\{([^{}]+)\}',lambda m:r'\path{'+m[1].replace(r'\_','_')+'}',t)
def raw(t):return '\n\n```{=latex}\n'+t+'\n```\n\n'

def tables(s,onecolumn=False):
    pattern=r'(?:\*\*Table ([A-Z0-9-]+) —([^\n]*)\n\s*)?((?:\|[^\n]*(?:\n|$))+)' 
    def convert(m):
        label,caption,block=m[1],m[2],m[3]
        lines=block.strip().splitlines()
        if len(lines)<3 or not re.match(r'^\|[\s:|\-]+\|$',lines[1]):return m[0]
        rows=[[c.strip() for c in line.strip().strip('|').split('|')] for line in [lines[0]]+lines[2:]]
        n=len(rows[0]);assert all(len(row)==n for row in rows),(label,rows)
        wide=not onecolumn and (n>=4 or label in ['I','II'])
        sizes=[]
        for j in range(n):
            lens=[len(re.sub(r'[$\\{}_*]','',r[j])) for r in rows]
            sizes.append(max(8,min(62,max(lens)*.6+sum(lens)/len(lens)*.4)))
        if n>=6:sizes[0]=max(sizes[0],13)
        if label=='I':sizes=[17,32,33,18]
        if label=='II':sizes=[20,34,46]
        if onecolumn and label=='S7':sizes=[14,7,30,5,12,12,13,12]
        if onecolumn and label=='S6':sizes=[6,8,13,16,18,17,17]
        if onecolumn and label=='S16':sizes=[42,10,24,24]
        weights=[x/sum(sizes)*n for x in sizes]
        cols=[]
        for j,w in enumerate(weights):
            align=r'\raggedright' if j==0 or label in ['I','II'] else (r'\raggedleft' if j==n-1 else r'\centering')
            cols.append('>{\\hsize='+f'{w:.6f}'+r'\hsize\linewidth=\hsize'+align+r'\arraybackslash}X')
        rr=[]
        for i,row in enumerate(rows):
            cells=[tex(c) for c in row]
            if i==0:cells=[r'\textbf{'+c+'}' for c in cells]
            rr.append(' & '.join(cells)+r' \\')
            if i==0:rr.append(r'\midrule')
        width=r'\textwidth' if wide or onecolumn else r'\columnwidth'
        body=r'\setlength{\tabcolsep}{3pt}'+'\n'+r'\renewcommand{\arraystretch}{1.12}'+'\n'+r'\begin{tabularx}{'+width+'}{@{}'+''.join(cols)+'@{}}\n'+r'\toprule'+'\n'+'\n'.join(rr)+'\n'+r'\bottomrule\end{tabularx}'
        if label:
            env='table*' if wide else 'table'
            cap=caption.replace('**','').strip()
            out=r'\begin{'+env+'}[!t]\n\\centering\n'+r'\renewcommand{\thetable}{'+label+'}\n\\caption{'+tex(cap)+'}\n'+body+'\n\\end{'+env+'}'
        else:out=r'\begin{center}'+body+r'\end{center}'
        return raw(out)
    return re.sub(pattern,convert,s)

def figures(s,main):
    # Figure captions occupy a single Markdown paragraph.
    pattern=r'!\[[^]]*\]\(([^)]+)\)\s*\*\*Fig\. ([A-Z0-9]+)\.\*\*\s*([^\n]+)'
    def convert(m):
        path,num,cap=m.groups();env='figure*' if main else 'figure'
        width=(r'\textwidth' if num=='3' else r'\textwidth') if main else (r'0.62\textwidth' if Path(path).stem not in ['fig2_preparation','fig_comfort_tail','fig_phase_cost'] else r'0.90\textwidth')
        file=Path(path).with_suffix('.pdf').name
        assert (P/'figs'/file).exists(),file
        return raw(r'\begin{'+env+'}[!t]\n\\centering\n'+r'\includegraphics[width='+width+']{'+file+'}\n'+r'\renewcommand{\thefigure}{'+num+'}\n\\caption{'+tex(cap)+'}\n\\end{'+env+'}')
    return re.sub(pattern,convert,s)

def headers(s,main):
    def convert(m):
        level,title=m.groups()
        if level=='##':
            if main:
                title=re.sub(r'^[IVX]+\. ','',title).title().replace('Td3','TD3')
                return raw(r'\section{'+tex(title)+'}')
            return raw(r'\FloatBarrier\stepcounter{section}\section*{'+tex(title)+'}'+r'\setcounter{subsection}{0}')
        if not main:return raw(r'\subsection*{'+tex(title)+'}')
        title=re.sub(r'^[A-Z](?:\.\d+)?\. ','',title)
        return raw(r'\subsection{'+tex(title)+'}')
    return re.sub(r'^(###?) (.+)$',convert,s,flags=re.M)

PRE=r'''\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb,graphicx,booktabs,array,tabularx,ragged2e,textcomp,url,xurl,cite,stfloats,placeins}
\usepackage[hidelinks]{hyperref}
\graphicspath{{../figs/}}
\urlstyle{same}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\setlength{\emergencystretch}{1.5em}
\hyphenation{rein-force-ment en-vi-ron-ment accel-er-a-tion}
\pagestyle{plain}
'''
def bib(refs,allowed=None):
    out=[r'\begin{thebibliography}{99}']
    entries=dict(re.findall(r'^- \*\*\[([^]]+)\]\*\* (.+)$',refs,re.M))
    for key in (list(entries) if allowed is None else allowed):
        out.append(r'\bibitem{'+key+'} '+tex(entries[key]))
    return '\n'.join(out+[r'\end{thebibliography}'])

def write(name,main):
    s=(P/(name+'.md')).read_text()
    refs=(P/'Manuscript_v27.md').read_text().split('## REFERENCES',1)[1].split('## AUTHOR BIOGRAPHIES')[0]
    title=(P/'Manuscript_v27.md').read_text().splitlines()[0][2:]
    if main:
        abstract=s.split('## ABSTRACT\n\n')[1].split('\n\n*Index Terms')[0]
        keywords=s.split('*Index Terms—')[1].split('*')[0].rstrip('.')
        s='## I. INTRODUCTION'+s.split('## I. INTRODUCTION',1)[1].split('## REFERENCES')[0]
        top=r'\title{'+tex(title)+'}\n'+r'\author{Ziran Peng and Zeyu Fan'+r'\thanks{Ziran Peng and Zeyu Fan are with the School of Transportation and Electrical Engineering, Hunan University of Technology, Zhuzhou, China (e-mail: pengziran@hut.edu.cn; m24085800005@stu.hut.edu.cn). Corresponding author: Ziran Peng.}}'+'\n'+r'\maketitle'+'\n'+r'\begin{abstract}'+tex(abstract)+r'\end{abstract}'+'\n'+r'\begin{IEEEkeywords}'+tex(keywords)+r'\end{IEEEkeywords}'
    else:
        s=s.split('Ziran Peng and Zeyu Fan',1)[1].strip()
        top=r'\title{Supplementary Material\\'+tex(title)+r'}\author{Ziran Peng and Zeyu Fan}\maketitle'
    allowed=list(dict.fromkeys(re.findall(r'\[((?:R-|S)[A-Za-z0-9-]+)\]',s)))
    s=figures(s,main);s=tables(s,not main);s=headers(s,main)
    content=tex(s)
    options='letterpaper,journal' if main else 'letterpaper,journal,onecolumn'
    biographies=''
    if main:
        bs=(P/'Manuscript_v27.md').read_text().split('## AUTHOR BIOGRAPHIES',1)[1]
        for who,body in re.findall(r'\*\*([^*]+)\*\* (.+)',bs):
            biographies+=r'\begin{IEEEbiographynophoto}{'+who+'}'+tex(body)+r'\end{IEEEbiographynophoto}'+'\n'
    t=r'\documentclass['+options+']{IEEEtran}\n'+PRE+'\n\\begin{document}\n'+top+'\n'+content+'\n'+bib(refs,allowed)+'\n'+biographies+'\n\\end{document}\n'
    (L/(name+'.tex')).write_text(t)
    for run in range(2):
        q=subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error','-file-line-error',name+'.tex'],cwd=L,capture_output=True,text=True,errors='replace')
        if q.returncode:
            print(q.stdout[-7500:]);raise RuntimeError(name+' compilation failed')
    shutil.copy2(L/(name+'.pdf'),P/(name+'.pdf'))
    log=(L/(name+'.log')).read_text(errors='replace')
    warnings=[line for line in log.splitlines() if any(w in line for w in ['Overfull','Missing character','undefined','Warning:'])]
    print(name,re.findall(r'Output written.*',log)[-1:]);print('\n'.join(warnings))
    return {'file':name+'.pdf','output':re.findall(r'Output written.*',log)[-1:],'warnings':warnings}

if __name__=='__main__':
    results=[write('Manuscript_v27',True),write('Supplementary_v27',False)]
    (P/'verification/build_log_summary.json').write_text(json.dumps(results,indent=2)+'\n')
