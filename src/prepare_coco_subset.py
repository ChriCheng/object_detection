import argparse,json,shutil
from pathlib import Path
from tqdm import tqdm

def args():
    p=argparse.ArgumentParser(); p.add_argument('--image_dir',required=True); p.add_argument('--ann_file',required=True); p.add_argument('--out_root',required=True); p.add_argument('--num_images',type=int,default=500); p.add_argument('--copy_mode',choices=['copy','symlink'],default='copy'); return p.parse_args()
def convert(b,w,h):
    x,y,bw,bh=b; return ((x+bw/2)/w,(y+bh/2)/h,bw/w,bh/h)
def main():
    a=args(); imgdir=Path(a.image_dir); ann=Path(a.ann_file); out=Path(a.out_root)
    if not imgdir.exists(): raise FileNotFoundError(imgdir)
    if not ann.exists(): raise FileNotFoundError(ann)
    data=json.loads(ann.read_text(encoding='utf-8'))
    imgs=sorted(data['images'],key=lambda x:x['file_name'])[:a.num_images]; ids={i['id'] for i in imgs}
    cats=sorted(data['categories'],key=lambda x:x['id']); cid={c['id']:i for i,c in enumerate(cats)}
    by={i:[] for i in ids}; anns=[]
    for an in data['annotations']:
        if an['image_id'] in ids: by[an['image_id']].append(an); anns.append(an)
    (out/'images/val2017').mkdir(parents=True,exist_ok=True); (out/'labels/val2017').mkdir(parents=True,exist_ok=True); (out/'annotations').mkdir(parents=True,exist_ok=True)
    lines=[]
    for im in tqdm(imgs,desc='prepare'):
        src=imgdir/im['file_name']; dst=out/'images/val2017'/im['file_name']
        if not src.exists(): continue
        if not dst.exists(): shutil.copy2(src,dst) if a.copy_mode=='copy' else dst.symlink_to(src.resolve())
        lines.append(f"./coco/images/val2017/{im['file_name']}")
        lab=[]
        for an in by.get(im['id'],[]):
            if an.get('iscrowd',0) or an.get('area',0)<=0: continue
            cls=cid.get(an['category_id']); xc,yc,bw,bh=convert(an['bbox'],im['width'],im['height'])
            if cls is not None and bw>0 and bh>0: lab.append(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
        (out/'labels/val2017'/(Path(im['file_name']).stem+'.txt')).write_text('\n'.join(lab)+('\n' if lab else ''),encoding='utf-8')
    subset={'info':data.get('info',{}),'licenses':data.get('licenses',[]),'images':imgs,'annotations':anns,'categories':data['categories']}
    (out/'annotations/instances_val2017.json').write_text(json.dumps(subset),encoding='utf-8')
    (out/'val2017.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (out/'train2017.txt').write_text('',encoding='utf-8'); (out/'test-dev2017.txt').write_text('',encoding='utf-8')
    print(f'[OK] {len(lines)} images, {len(anns)} annotations')
if __name__=='__main__': main()
