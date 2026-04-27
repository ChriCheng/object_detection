from pathlib import Path
root=Path(__file__).resolve().parents[1]
for name,p in {'MindYOLO':root/'third_party/mindyolo','weight':root/'weights/yolov8n.ckpt','raw images':root/'data/raw/coco/val2017','raw annotation':root/'data/raw/coco/annotations/instances_val2017.json','val txt':root/'data/coco_mindyolo/coco/val2017.txt'}.items(): print(f"[{'OK' if p.exists() else 'MISS'}] {name}: {p}")
