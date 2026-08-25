#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Normalize reasoning_steps in JSONL files so that each record keeps only the last
(combined) step as a single-element list.

- Input files are read line by line (memory-safe for large files).
- If `reasoning_steps` is a non-empty list, it becomes `[reasoning_steps[-1]]`.
- If missing or not a list or empty, it's left unchanged.
- Other fields are preserved as-is.

Usage (from repo root or this script directory):
  python scripts/normalize_reasoning_steps.py \
    "Rethinking Tabular DeepSeek new/buzhou/training_data_zhengti_llama/negative_samples_standardized.jsonl" \
    "Rethinking Tabular DeepSeek new/buzhou/training_data_zhengti_llama/positive_samples_standardized.jsonl"

This writes two output files:
  negative_samples_standardized.jsonl (overwrite)
  positive_samples_standardized.jsonl (overwrite)
"""
from __future__ import annotations
import sys
import os
import json
from typing import Any, Dict


def normalize_file(input_path: str) -> str:
    dir_path = os.path.dirname(input_path)
    base = os.path.basename(input_path)
    # Write to temp file first
    temp_path = input_path + '.tmp'
    output_path = input_path

    with open(input_path, 'r', encoding='utf-8') as fin, open(temp_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            line = line.rstrip('\n')
            if not line:
                fout.write('\n')
                continue
            try:
                obj: Dict[str, Any] = json.loads(line)
            except Exception:
                # preserve raw line if not valid json
                fout.write(line + '\n')
                continue

            steps = obj.get('reasoning_steps', None)
            if isinstance(steps, list) and len(steps) > 0:
                # keep only the last, as a single-element list
                last = steps[-1]
                obj['reasoning_steps'] = [last]
            # else: leave unchanged

            fout.write(json.dumps(obj, ensure_ascii=False) + '\n')

    # Replace original with temp file
    os.replace(temp_path, output_path)
    return output_path


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(
            'Usage: python scripts/normalize_reasoning_steps.py <file1.jsonl> [file2.jsonl ...]')
        return 1

    for path in argv[1:]:
        out = normalize_file(path)
        print(f'Wrote: {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
