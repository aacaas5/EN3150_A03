"""Read all experiment CSVs and fingerprint immutable report evidence."""
from pathlib import Path
import hashlib
import json
import pandas as pd
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'report/audit'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def run():
    paths = sorted((ROOT / 'results').rglob('*.csv'))
    summary = {}
    for path in paths:
        frame = pd.read_csv(path)
        key = path.relative_to(ROOT).as_posix()
        summary[key] = {'rows': len(frame), 'columns': list(frame.columns), 'sha256': digest(path)}
        print(key, frame.shape)
        if path.name == 'test_predictions.csv':
            print('  correct:', int((frame.true_label == frame.predicted_label).sum()))
        elif path.name == 'history.csv':
            print(frame[['epoch','train_loss','val_loss','train_accuracy','val_accuracy']].to_string(index=False))
        elif 'architecture' in path.name:
            print(frame.groupby('type')[['parameters','conv_linear_macs']].sum().to_string())
        else:
            print(frame.to_string(index=False))
    evidence = [p for p in (ROOT / 'results').rglob('*') if p.is_file()]
    evidence += [p for folder in ('src','scripts') for p in (ROOT / folder).rglob('*.py')]
    evidence += [ROOT/'config.py',ROOT/'README.md',ROOT/'RUN_LOG.md']
    baseline = OUT / 'evidence_hashes_before.json'
    if not baseline.exists():
        baseline.write_text(json.dumps({p.relative_to(ROOT).as_posix():digest(p) for p in evidence}, indent=2))
    (OUT/'csv_inventory.json').write_text(json.dumps(summary,indent=2))
    # Read every generated plot. PDF/PNG variants show identical saved figures.
    images = sorted((ROOT/'results').rglob('*.png'))
    for start in range(0,len(images),6):
        canvas = Image.new('RGB',(1600,1500),'#eeeeee')
        draw = ImageDraw.Draw(canvas)
        for j,path in enumerate(images[start:start+6]):
            im = Image.open(path).convert('RGB')
            im.thumbnail((780,465))
            x,y = (j%2)*800,(j//2)*500
            canvas.paste(im,(x+(800-im.width)//2,y+25))
            draw.text((x+12,y+5),path.relative_to(ROOT).as_posix(),fill='black')
        canvas.save(OUT/f'plots-{start//6+1}.png')

if __name__ == '__main__':
    run()
