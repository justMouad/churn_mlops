"""python scripts/train_all.py [--quick] [--out artifacts]"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from churnops.config import ARTIFACTS_DIR  # noqa: E402
from churnops.pipeline import run_training  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="entraînement allégé (tests / CI)")
    ap.add_argument("--out", default=str(ARTIFACTS_DIR))
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    reg = run_training(Path(a.out), quick=a.quick, seed=a.seed)
    print("Version :", reg["version"])
