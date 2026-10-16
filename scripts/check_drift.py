"""Contrôle de dérive des données.

Compare un lot de données récentes à la référence d'entraînement.
  python scripts/check_drift.py --current data/incoming/latest.csv
  python scripts/check_drift.py --simulate-drift          # démo : lot synthétique décalé
Sortie : reports/drift_report.json ; si $GITHUB_OUTPUT existe, écrit retrain=true|false.
Code retour 0 (utiliser --fail-on-alert pour échouer sur une dérive forte).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from churnops.config import ARTIFACTS_DIR, CATEGORICAL, NUMERIC, REPORTS_DIR  # noqa: E402
from churnops.data import make_customers  # noqa: E402
from churnops.monitoring import drift_report  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--current", help="CSV de données récentes (mêmes colonnes que la base clients)")
    ap.add_argument("--simulate-drift", action="store_true")
    ap.add_argument("--fail-on-alert", action="store_true")
    a = ap.parse_args()

    ref = pd.read_csv(ARTIFACTS_DIR / "churn_reference.csv")
    if a.current and Path(a.current).exists():
        cur = pd.read_csv(a.current)
    else:
        print("Aucun fichier fourni/trouvé : lot synthétique", "décalé" if a.simulate_drift else "stable")
        cur = make_customers(3000, seed=123, drift=a.simulate_drift)
    report = drift_report(ref, cur, NUMERIC, CATEGORICAL)

    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "drift_report.json").write_text(json.dumps(report, indent=2))
    print(f"Statut global : {report['overall_status']} | alertes={report['n_alert']} "
          f"avertissements={report['n_warning']} | réentraînement recommandé : {report['retrain_recommended']}")
    for name, f in list(report["features"].items())[:5]:
        print(f"  {name:24s} PSI={f['psi']:.3f}  {f['status']}")
    if os.getenv("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as fh:
            fh.write(f"retrain={'true' if report['retrain_recommended'] else 'false'}\n")
    sys.exit(1 if (a.fail_on_alert and report["overall_status"] == "alert") else 0)
