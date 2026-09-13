"""把 reports/_v4_tables.json 里的表格填进 REPORT_v4_zh.md 的“（表：key）”占位。用法：python visual_dev/v4_report_draft.py && python visual_dev/v4_fill_report.py"""
import json, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; t = json.load(open(ROOT / 'reports/_v4_tables.json')); p = ROOT / 'reports/REPORT_v4_zh.md'; s = p.read_text()
def rep(m):
    k = m.group(1)
    if k == 'agg' and t.get('agg'): return t['agg_hdr'] + "\n" + t['agg']
    if k == 'paired' and t.get('paired'): return t['paired_hdr'] + "\n" + t['paired']
    if k == 'cm' and t.get('cm'): return t['cm_hdr'] + "\n" + t['cm']
    if k == 'failures' and t.get('failures'): return t['fail_hdr'] + "\n" + t['failures']
    return (t['hdr'] + "\n" + t[k]) if t.get(k) else m.group(0)
s2 = re.sub(r'（表：(\w+)）', rep, s); p.write_text(s2); print('已填表：', [k for k in re.findall(r'（表：(\w+)）', s) if t.get(k) or k in ('agg', 'paired')], '仍占位：', re.findall(r'（表：(\w+)）', s2))
