"""Read-only verification of experimental evidence and revised report numbers.

Only audit output and report provenance are written, both within report/.
The original experiment verification record remains untouched.
"""
from pathlib import Path
import hashlib
import json
import re
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from build_numeric_assets import assets, NAMES

ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'report'
checks=[]

def check(name, ok):
    assert ok, name
    checks.append(name)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def run():
    baseline=json.loads((REPORT/'audit/evidence_hashes_before.json').read_text())
    check('All original result files, training/evaluation code, README and run log unchanged',
          all(digest(ROOT/p)==value for p,value in baseline.items()))
    for relative, expected in assets().items():
        check('Report numbers: '+relative,(REPORT/relative).read_text()==expected)
    final=pd.read_csv(ROOT/'results/tables/final_comparison.csv').set_index('model')
    indices=json.loads((ROOT/'data/splits/test_indices.json').read_text())
    for n in NAMES:
        raw=ROOT/f'results/raw/{n}'
        predictions=pd.read_csv(raw/'test_predictions.csv')
        check(n+' exact test samples',predictions.sample_index.tolist()==indices)
        y,p=predictions.true_label,predictions.predicted_label
        values={'test_accuracy':accuracy_score(y,p),
                'macro_precision':precision_score(y,p,average='macro',labels=range(10),zero_division=0),
                'macro_recall':recall_score(y,p,average='macro',labels=range(10),zero_division=0)}
        for key,val in values.items():
            check(n+' '+key,abs(val-final.loc[n,key])<1e-12)
        check(n+' numeric confusion matrix',np.array_equal(confusion_matrix(y,p,labels=range(10)),pd.read_csv(raw/'confusion_matrix.csv',index_col=0).values))
        meta=json.loads((raw/'run.json').read_text())
        history=pd.read_csv(raw/'history.csv')
        check(n+' epoch count and best checkpoint',len(history)==meta['epochs'] and history.val_accuracy.idxmax()+1==meta['best_epoch'])
        check(n+' parameter sum',pd.read_csv(ROOT/f'results/tables/{n}_architecture.csv').parameters.sum()==final.loc[n,'total_parameters'])
    # Remaining literal experimental numbers in prose: exact confusion entries,
    # split counts and the epoch of the largest custom validation decrease.
    for n,true,pred,count in [('model_a','HerbaceousVegetation','PermanentCrop',50),
          ('model_b','Highway','Industrial',22),('mobilenet_v2','PermanentCrop','HerbaceousVegetation',10),
          ('mobilenet_v2','River','Highway',10),('efficientnet_b0','AnnualCrop','PermanentCrop',14)]:
        cm=pd.read_csv(ROOT/f'results/raw/{n}/confusion_matrix.csv',index_col=0)
        check(n+' prose confusion count '+true,int(cm.loc[true,pred])==count)
    for n in ('model_a','model_b'):
        h=pd.read_csv(ROOT/f'results/raw/{n}/history.csv')
        check(n+' prose validation drop epoch',h.val_accuracy.diff().idxmin()+1==11)
    info=json.loads((ROOT/'data/splits/dataset_info.json').read_text())
    check('Report split sizes',info['split_sizes']=={'train':18900,'val':4050,'test':4050})
    check('Model B meets limit',final.loc['model_b','trainable_parameters']==22426 and final.loc['model_b','trainable_parameters']<100000)
    main=(REPORT/'main.tex').read_text()
    section_files=list((REPORT/'sections').glob('*.tex'))
    text=main+'\n'+'\n'.join(p.read_text() for p in section_files)
    expanded=text
    for path in (REPORT/'tables').glob('*.tex'):
        expanded=expanded.replace('\\input{tables/'+path.stem+'}',path.read_text())
    labels=re.findall(r'\\label\{([^}]+)\}',expanded)
    refs=re.findall(r'\\(?:eqref|ref)\{([^}]+)\}',expanded)
    check('Unique labels',len(labels)==len(set(labels)))
    check('All references defined',set(refs)<=set(labels))
    floats=re.findall(r'\\begin\{(figure|table|longtable)\}(.*?)\\end\{\1\}',expanded,re.S)
    for kind,body in floats:
        label=re.search(r'\\label\{([^}]+)\}',body)
        check(kind+' caption and label',r'\caption{' in body and label is not None)
        check(kind+' referenced in prose: '+label.group(1),label.group(1) in refs)
    bib=(REPORT/'references.bib').read_text()
    keys=set(re.findall(r'@\w+\{([^,]+),',bib))
    citations={k for group in re.findall(r'\\cite\{([^}]+)\}',expanded) for k in group.split(',')}
    check('All citation keys in bibliography',citations<=keys)
    for figure in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',expanded):
        path=REPORT/figure
        originals=list((ROOT/'results').rglob(path.name))
        check('Original figure unmodified: '+path.name,any(digest(path)==digest(p) for p in originals))
    log=(REPORT/'main_revised.log').read_text(errors='replace')
    blg=(REPORT/'main_revised.blg').read_text(errors='replace')
    warnings=[line for line in log.splitlines() if re.search(r'(LaTeX|Package .*|pdfTeX) [Ww]arning|Overfull|Underfull|^!',line)]
    check('No LaTeX errors, unresolved citations, layout or PDF warnings',not warnings and 'undefined' not in log.lower())
    check('No BibTeX warnings', 'Warning--' not in blg)
    # Update report provenance only after validating numbers and immutability.
    provenance=json.loads((REPORT/'result_provenance.json').read_text())
    for path in [REPORT/'main.tex',REPORT/'references.bib',*section_files,*list((REPORT/'tables').glob('*.tex'))]:
        provenance[path.relative_to(ROOT).as_posix()]=digest(path)
    (REPORT/'result_provenance.json').write_text(json.dumps(provenance,indent=2))
    output={'passed':True,'checks':checks,'unresolved_latex_warnings':warnings,
            'pdf':'report/main_revised.pdf','pdf_sha256':digest(REPORT/'main_revised.pdf'),
            'experimental_results_unchanged':True}
    (REPORT/'audit/verification.json').write_text(json.dumps(output,indent=2))
    print('PASS:',len(checks),'report checks; all experiment evidence unchanged.')

if __name__=='__main__':
    run()
