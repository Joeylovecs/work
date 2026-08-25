import json
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(sys.argv[1]) if len(sys.argv)>1 else Path('outputs/50_cycle2')
OUT = ROOT / 'RESULT_INDEX.md'

def rows(path):
    return [json.loads(x) for x in (path/'result.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]

def paired_metrics(ds):
    base={r['sample_id']:r for r in rows(ROOT/ds/'baseline')}
    aud={r['sample_id']:r for r in rows(ROOT/ds/'audit')}
    silent=[sid for sid,r in base.items() if r.get('execution_success') and not r.get('correct')]
    detected=[sid for sid in silent if aud.get(sid,{}).get('semantic_exception')]
    repaired=[sid for sid in detected if aud[sid].get('correct')]
    false_flags=[sid for sid,r in base.items() if r.get('correct') and aud.get(sid,{}).get('semantic_exception')]
    degradation=[sid for sid,r in base.items() if r.get('correct') and not aud.get(sid,{}).get('correct')]
    return silent,detected,repaired,false_flags,degradation

lines=['# 实验结果索引','',f'- 根目录：`{ROOT}`','- 区间：由 `selected_samples.json` 确定；当前实验默认是 flat index `[0, 100)`，end 为开区间。','- 详细逐条结果：各方法目录下的 `result.jsonl`；逐条第一篇兼容日志：`log/<flat_index>.txt`。','']
for ds in ('wtq','tabfact'):
    lines.append(f'## {ds.upper()}')
    lines.append('')
    for method in ('baseline','audit','joint'):
        p=ROOT/ds/method
        s=json.loads((p/'summary.json').read_text(encoding='utf-8'))
        lines.append(f'### {method}')
        lines.append(f"- accuracy: {s['accuracy']:.2%}; execution success: {s['execution_success_rate']:.2%}; execution-success-but-wrong: {s['execution_success_but_wrong']}; API calls: {s['api_calls']}")
        if method=='audit':
            lines.append(f"- semantic exceptions: {s['semantic_exception_count']}; layers: {s['layer_error_counts']}; error types: {s['error_type_counts']}")
        lines.append(f'- 文件：`{p}/summary.json`、`{p}/result.jsonl`、`{p}/selected_samples.json`')
        lines.append('')
    silent,detected,repaired,false_flags,degradation=paired_metrics(ds)
    lines.append('### 配对审计统计')
    lines.append(f'- baseline 静默错误候选：{len(silent)}')
    lines.append(f'- auditor 检测到：{len(detected)}；检测率：{len(detected)/len(silent):.2%}' if silent else '- auditor 检测到：0')
    lines.append(f'- semantic repair 后恢复正确：{len(repaired)}；false positive flags：{len(false_flags)}；repair degradation：{len(degradation)}')
    lines.append('')
    if silent:
        lines.append('- silent IDs：' + ', '.join(silent))
        lines.append('')

OUT.write_text('\n'.join(lines)+'\n', encoding='utf-8')
print(OUT)
