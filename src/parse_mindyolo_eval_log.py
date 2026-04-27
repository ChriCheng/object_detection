import argparse,json,re
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--log_file',required=True); p.add_argument('--out_file',required=True); a=p.parse_args()
t=Path(a.log_file).read_text(encoding='utf-8',errors='ignore')
def f(pat):
    m=re.search(pat,t,re.I|re.S); return float(m.group(1)) if m else None
m={'source_log':a.log_file,'precision':None,'recall':None,'map_50':f(r'Average Precision\s*\(AP\).*?IoU=0\.50\s+.*?=\s*([0-9.]+)'),'map_50_95':f(r'Average Precision\s*\(AP\).*?IoU=0\.50:0\.95.*?=\s*([0-9.]+)'),'fps':f(r'FPS\s*[:=]\s*([0-9.]+)')}
tab=re.search(r'all\s+\d+\s+\d+\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)',t)
if tab: m.update({'precision':float(tab.group(1)),'recall':float(tab.group(2)),'map_50':m['map_50'] or float(tab.group(3)),'map_50_95':m['map_50_95'] or float(tab.group(4))})
out=Path(a.out_file); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(m,indent=2,ensure_ascii=False),encoding='utf-8'); print(json.dumps(m,indent=2,ensure_ascii=False))
