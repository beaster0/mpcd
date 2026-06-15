#!/usr/bin/env python3
from __future__ import annotations
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path('/home/zhangxh/students/sjl/mpcd_modular_v2')
PY = '/home/zhangxh/students/sjl/miniconda3/envs/A/bin/python'
LOG = ROOT / 'logs' / 'fluid_gpu3.log'
TASKS = [(g, seed) for g in ['0','0.001','0.003','0.005','0.01'] for seed in [301,302,303]]

def write(msg: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(msg + '\n')
        f.flush()

def now() -> str:
    return datetime.now().strftime('%F_%T')

write(f'[start] {now()} gpu3 fluid production python scheduler')
for g, seed in TASKS:
    cmd = [PY, 'scripts/??.py', '--structure', 'fluid', '--force', g, '--seed', str(seed), '--gpu', '3', '--steps', '1000000']
    write(f'[task] {now()} ' + ' '.join(cmd))
    with LOG.open('a', encoding='utf-8') as f:
        result = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    write(f'[done] {now()} fluid g={g} seed={seed} code={result.returncode}')
    if result.returncode != 0:
        raise SystemExit(result.returncode)
write(f'[finish] {now()} gpu3 fluid production python scheduler')
