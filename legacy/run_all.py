# -*- coding: utf-8 -*-
import os
import subprocess
import sys

base = os.path.dirname(__file__)
steps = [
    'step1_preprocess.py',
    'step2_denoise.py',
    'step3_risk_model.py',
    'step4_visualize_report.py',
]

for s in steps:
    print('=' * 60)
    print('RUN', s)
    print('=' * 60)
    subprocess.check_call([sys.executable, os.path.join(base, s)], cwd=base)
