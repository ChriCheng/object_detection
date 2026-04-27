import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]; p=root/'results/metrics/yolov8n_eval_result.json'
m=json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
f=lambda x:'以实际日志为准' if x is None else str(x)
(root/'results/metrics/summary.md').write_text(f"""# YOLOv8n 评估结果汇总\n\n| 模型 | 数据集 | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | FPS |\n|---|---|---:|---:|---:|---:|---:|\n| YOLOv8n | COCO val2017/subset | {f(m.get('precision'))} | {f(m.get('recall'))} | {f(m.get('map_50'))} | {f(m.get('map_50_95'))} | {f(m.get('fps'))} |\n""",encoding='utf-8')
