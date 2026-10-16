"""Garde-fou de promotion : le modèle candidat ne doit pas être pire que celui en production.

  python scripts/promote_gate.py --prod /tmp/prev_registry.json --candidate artifacts/registry.json
Code retour 1 si régression (PR-AUC churn > 0,02 de moins, ou MAPE prévision > 10 % pire).
"""
import argparse
import json
import sys
from pathlib import Path

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prod", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--pr-auc-tol", type=float, default=0.02)
    ap.add_argument("--mape-tol", type=float, default=0.10)
    a = ap.parse_args()
    cand = json.loads(Path(a.candidate).read_text())
    if not Path(a.prod).exists():
        print("Pas de modèle de référence : promotion acceptée.")
        sys.exit(0)
    prod = json.loads(Path(a.prod).read_text())
    ok = True
    d = cand["churn"]["pr_auc"] - prod["churn"]["pr_auc"]
    print(f"Churn PR-AUC : prod={prod['churn']['pr_auc']:.3f} candidat={cand['churn']['pr_auc']:.3f} (Δ={d:+.3f})")
    if d < -a.pr_auc_tol:
        ok = False
        print("  ✗ régression churn")
    r = cand["forecast"]["mape"] / prod["forecast"]["mape"]
    print(f"Prévision MAPE : prod={prod['forecast']['mape']:.2f}% candidat={cand['forecast']['mape']:.2f}% (x{r:.2f})")
    if r > 1 + a.mape_tol:
        ok = False
        print("  ✗ régression prévision")
    print("Promotion", "ACCEPTÉE" if ok else "REFUSÉE")
    sys.exit(0 if ok else 1)
