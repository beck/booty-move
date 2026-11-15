from pathlib import Path
from PIL import Image
import cv2
import numpy as np
import subprocess, tempfile, xml.etree.ElementTree as ET

root=Path(__file__).resolve().parent
source=root.parent/'ChatGPT Image Nov 15, 2026, 07_54_04 AM.png'
im=np.array(Image.open(source).convert('RGB'))
h,w=im.shape[:2]
# Edge-preserving denoising suppresses paper grain while retaining printed dots.
smooth=cv2.bilateralFilter(im,5,28,3)
prototypes=np.array([[252,194,10],[242,197,158],[226,79,43],[35,75,107],[45,36,30]],np.uint8)
lab=cv2.cvtColor(smooth,cv2.COLOR_RGB2LAB).astype(np.float32)
pal=cv2.cvtColor(prototypes[None,:,:],cv2.COLOR_RGB2LAB)[0].astype(np.float32)
labels=np.argmin(((lab[:,:,None,:]-pal[None,None,:,:])**2).sum(axis=3),axis=2).astype(np.uint8)
# Remove isolated texture specks without changing larger halftone marks.
med=cv2.medianBlur(labels,3)
for i in range(5):
    n,cc,stats,_=cv2.connectedComponentsWithStats((labels==i).astype(np.uint8),8)
    tiny=np.flatnonzero(stats[:,cv2.CC_STAT_AREA]<5)
    tiny=tiny[tiny!=0]
    mask=np.isin(cc,tiny)
    labels[mask]=med[mask]
# Recover skin regions from the small red halftone components.
n, cc, stats, _ = cv2.connectedComponentsWithStats((labels==2).astype(np.uint8),8)
ids=np.flatnonzero((stats[:,cv2.CC_STAT_AREA]>=4)&(stats[:,cv2.CC_STAT_AREA]<110))
ids=ids[ids!=0]
seeds=np.isin(cc,ids).astype(np.uint8)
skin=cv2.morphologyEx(seeds,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15)))
# Keep substantial skin regions, discarding isolated red details.
n, regions, stats, _=cv2.connectedComponentsWithStats(skin,8)
ids=np.flatnonzero(stats[:,cv2.CC_STAT_AREA]>1000); ids=ids[ids!=0]
skin=np.isin(regions,ids)&np.isin(labels,[1,2])
labels[skin]=1
colors=[('Yellow background','#fcc20a'),('Cream paper and highlights','#f2c59e'),('Red ink and halftone','#e24f2b'),('Blue jacket','#234b6b'),('Black outlines and lettering','#2d241e')]
ns='http://www.w3.org/2000/svg'
ink='http://www.inkscape.org/namespaces/inkscape'
# Inkscape's actual namespace.
ink='http://www.inkscape.org/namespaces/inkscape'.replace('www.inkscape.org/namespaces','www.inkscape.org/namespaces')
ET.register_namespace('',ns); ET.register_namespace('inkscape',ink)
def svgroot():
    return ET.Element('{%s}svg'%ns,{'width':str(w),'height':str(h),'viewBox':f'0 0 {w} {h}'})
svg=svgroot()
ET.SubElement(svg,'{%s}title'%ns).text='Booty Move — cleaned five-color vector'
ET.SubElement(svg,'{%s}desc'%ns).text='Original artwork traced into editable paths. Each ink color is a separate named Inkscape layer. No embedded raster images or font dependencies.'
with tempfile.TemporaryDirectory() as td:
    skin_pbm=Path(td)/'skin.pbm'
    skin_svg=Path(td)/'skin.svg'
    Image.fromarray(np.where(skin,0,255).astype(np.uint8)).convert('1').save(skin_pbm)
    subprocess.run(['potrace',str(skin_pbm),'-s','-o',str(skin_svg),'--turdsize','4','--opttolerance','0.2'],check=True)
    defs=ET.SubElement(svg,'{%s}defs'%ns)
    pattern=ET.SubElement(defs,'{%s}pattern'%ns,{'id':'uniform-dots','patternUnits':'userSpaceOnUse','width':'8','height':'8'})
    for cx,cy in [(2,2),(6,6)]:
        ET.SubElement(pattern,'{%s}circle'%ns,{'cx':str(cx),'cy':str(cy),'r':'1.65','fill':colors[2][1]})
    clip=ET.SubElement(defs,'{%s}clipPath'%ns,{'id':'skin-clip','clipPathUnits':'userSpaceOnUse'})
    for g in ET.parse(skin_svg).getroot().findall('{%s}g'%ns):
        clip.append(g)

    for i,(name,color) in enumerate(colors):
        layer=ET.SubElement(svg,'{%s}g'%ns,{'id':f'layer-{i+1}','{%s}groupmode'%ink:'layer','{%s}label'%ink:name,'fill':color})
        if i==0:
            ET.SubElement(layer,'{%s}rect'%ns,{'width':str(w),'height':str(h)})
        else:
            pbm=Path(td)/f'{i}.pbm'; traced=Path(td)/f'{i}.svg'
            Image.fromarray(np.where(labels==i,0,255).astype(np.uint8)).convert('1').save(pbm)
            subprocess.run(['potrace',str(pbm),'-s','-o',str(traced),'--turdsize','4','--alphamax','1','--opttolerance','0.2'],check=True)
            tree=ET.parse(traced)
            for g in tree.getroot().findall('{%s}g'%ns):
                g.attrib.pop('fill',None)
                layer.append(g)
        if i==2:
            dots=ET.SubElement(layer,'{%s}g'%ns)
            interior=cv2.distanceTransform(skin.astype(np.uint8),cv2.DIST_L2,5)
            for y in range(2,h,4):
                for x in range(2+((y//4)%2)*4,w,8):
                    if interior[y,x]>=1.9:
                        ET.SubElement(dots,'{%s}circle'%ns,{'cx':str(x),'cy':str(y),'r':'1.65'})
        sep=svgroot()
        import copy
        sep.append(copy.deepcopy(defs))
        import copy
        sep.append(copy.deepcopy(layer))
        ET.ElementTree(sep).write(root/f'codex-{i+1:02d}-{name.lower().replace(" ","-")}.svg',encoding='utf-8',xml_declaration=True)
ET.ElementTree(svg).write(root/'codex-booty-move-layered.svg',encoding='utf-8',xml_declaration=True)
print(f'Created layered SVG at {w} x {h}, with five color separations.')
