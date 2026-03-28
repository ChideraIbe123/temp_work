"""Run all ablation tests and print comparison table."""
import subprocess, sys, os

tests = [
    "0_full.py",
    "1_no_quantile.py",
    "2_no_baseline.py",
    "3_ml_abandon.py",
    "4_no_bias.py",
    "5_flat_bias.py",
]

dir = os.path.dirname(os.path.abspath(__file__))
for t in tests:
    print(f"\n{'='*60}")
    path = os.path.join(dir, t)
    subprocess.run([sys.executable, path], cwd=dir)
