from pathlib import Path
from PIL import Image, ImageDraw

folder=Path(__file__).resolve().parent
files=sorted((folder/'pages').glob('page-*.png'))
for start in range(0,len(files),9):
    canvas=Image.new('RGB',(1260,1830),'#dddddd')
    for j,path in enumerate(files[start:start+9]):
        im=Image.open(path).convert('RGB');im.thumbnail((410,580))
        x,y=(j%3)*420,(j//3)*610
        canvas.paste(im,(x,y+24))
        ImageDraw.Draw(canvas).text((x+10,y+5),path.stem,fill='black')
    canvas.save(folder/f'review-{start//9+1}.png')
