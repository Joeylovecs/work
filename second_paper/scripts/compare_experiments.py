import argparse, json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from second_paper.evaluation.metrics import compare

def load(path): return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
ap=argparse.ArgumentParser(); ap.add_argument("baseline"); ap.add_argument("audited"); ap.add_argument("--output",required=True); args=ap.parse_args()
result=compare(load(args.baseline),load(args.audited)); Path(args.output).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps(result,ensure_ascii=False,indent=2))
