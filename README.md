# Prévision & prédiction du churn — pipeline MLOps

Deux modèles servis par **FastAPI**, conteneurisés avec **Docker**, avec tests / réentraînement / déploiement
automatisés par **GitHub Actions** et un suivi de base de la **dérive** des données et des modèles.

| Brique | Contenu |
|---|---|
| Prévision | Série quotidienne de clients actifs : baseline saisonnière, **Prophet**, **LSTM (PyTorch)**, backtest à origine glissante |
| Churn | Régression logistique vs Gradient Boosting (grille CV sur PR-AUC), seuil optimisé sur le F1 |
| API | `/predict/churn`, `/predict/churn/batch`, `/forecast`, `/monitoring/*`, `/health`, `/model/info` |
| Monitoring | PSI + KS par variable, dérive des prédictions, dégradation du MAPE et de l'AUC |
| CI/CD | `ci.yml` (lint, tests, build + smoke test Docker), `retrain.yml` (dérive → réentraînement → garde-fou → PR), `deploy.yml` (GHCR + SSH optionnel) |

## Démarrage rapide

```bash
pip install -r requirements.txt -r requirements-lstm.txt -r requirements-dev.txt
make test                       # 19 tests
make train                      # réentraîne et écrit artifacts/ (≈ 1 min)
make run                        # http://127.0.0.1:8000/docs
make drift                      # démo de détection de dérive
docker compose up --build       # ou: make docker-build docker-run
```

Les `artifacts/` fournis sont déjà entraînés : l'API et l'image Docker fonctionnent immédiatement.

## Exemples d'appels

```bash
curl -X POST localhost:8000/predict/churn -H 'Content-Type: application/json' -d '{
  "customer_id":"C-1001","tenure_months":5,"monthly_charges":92.5,"contract_type":"month-to-month",
  "payment_method":"electronic_check","support_tickets_90d":4,"usage_hours_30d":22,
  "late_payments_6m":2,"num_products":1,"has_discount":0,"age":34}'
# -> {"churn_probability":0.96,"churn_predicted":true,"risk_tier":"high","threshold":0.29,...}

curl -X POST localhost:8000/forecast -H 'Content-Type: application/json' -d '{"horizon":14,"model":"champion"}'
curl localhost:8000/monitoring/drift          # trafic réel récent vs référence d'entraînement
```

## Résultats (données synthétiques)

**Prévision, backtest 3 x 28 jours**

| Modèle | MAPE | sMAPE | MAE |
|---|---|---|---|
| Baseline saisonnière (7 j) | 6,72 % | 6,42 % | 484 |
| **Prophet (champion)** | **3,24 %** | 3,26 % | 240 |
| LSTM | 5,39 % | 5,23 % | 386 |

Prophet gagne ici : série avec tendance linéaire, saisonnalités nettes et peu de données (3 ans). Un LSTM n'est pas
automatiquement meilleur ; la baseline sert de garde-fou. Voir `reports/forecast_comparison.png`.

**Churn (test, 20 % de churn)** : régression logistique retenue, ROC-AUC 0,81, PR-AUC 0,51 (hasard = 0,20),
rappel 0,63, précision 0,46, seuil 0,29. Le signal du jeu synthétique est surtout linéaire, d'où la victoire de la
logistique sur le boosting.

## Monitoring : comment ça marche

- **Dérive des variables** : PSI par variable (bornes en quantiles de la référence ; variables binaires traitées par
  modalité), plus test KS. Seuils : < 0,10 stable, 0,10-0,25 à surveiller, > 0,25 alerte. Réentraînement recommandé
  si ≥ 1 alerte ou beaucoup d'avertissements.
- **Dérive des prédictions** : PSI sur `churn_probability` (inclus dans les rapports).
- **Trafic réel** : l'API garde en mémoire les 5 000 dernières requêtes ; `GET /monitoring/drift` les compare à la référence.
- **Performance** : `POST /monitoring/forecast-performance` compare les valeurs observées aux prévisions (MAPE vs backtest,
  alerte à x1,5) ; `churnops.monitoring.churn_performance` suit la perte d'AUC quand les vrais labels arrivent.
- Sur le lot synthétique décalé, 3 variables passent en alerte (`monthly_charges` PSI 0,51, `support_tickets_90d`
  0,43, `tenure_months` 0,28) ; sur un lot stable, aucune.

## Automatisation GitHub Actions

```
push / PR ──► ci.yml : ruff → pytest → build Docker → smoke test (/health, /forecast)
cron lundi ─► retrain.yml : check_drift ─(dérive ou "force")─► train_all ─► pytest ─► promote_gate ─► PR artefacts
merge main ─► deploy.yml : build + push ghcr.io/<repo>:latest ─► déploiement SSH (si DEPLOY_ENABLED=true)
```

`scripts/promote_gate.py` refuse un candidat dont la PR-AUC churn baisse de plus de 0,02 ou dont le MAPE se dégrade
de plus de 10 % par rapport au modèle en production.

**À brancher pour la production** : l'étape « Récupérer les données récentes » de `retrain.yml` (S3, base de
données, entrepôt) ; sans fichier `data/incoming/latest.csv`, un lot synthétique est utilisé. Pour le déploiement :
variable `DEPLOY_ENABLED=true` et secrets `SSH_HOST`, `SSH_USER`, `SSH_KEY`.

## Structure

```
churnops/  config, data (générateurs), forecasting, churn, monitoring, pipeline (entraînement complet)
app/       main.py (endpoints), schemas.py (Pydantic), state.py (chargement artefacts + tampon live)
scripts/   train_all, check_drift, promote_gate, make_report
tests/     API, monitoring, modèles (Prophet/LSTM aller-retour de sérialisation)
artifacts/ modèles + référence + registry.json (versionnés dans le dépôt pour la démo)
.github/workflows/  ci, retrain, deploy      Dockerfile · docker-compose.yml
```

## Limites

- Données **synthétiques** ; chiffres illustratifs. Remplacer `churnops/data.py` par vos sources.
- Tampon de requêtes **en mémoire** (perdu au redémarrage, par instance) : en production, journaliser dans une base
  ou un outil dédié (Evidently, Prometheus/Grafana).
- Artefacts versionnés dans git par simplicité ; en production, préférer un registre de modèles (MLflow, S3).
- Dockerfile et workflows **non exécutés dans mon environnement** (pas de Docker ni de runner GitHub) : syntaxe YAML
  validée, API et scripts testés en local. Premier build : `docker compose up --build`.
- LSTM sans intervalles de prédiction (Prophet en fournit). Le conteneur fonctionne sans PyTorch (`WITH_LSTM=false`,
  repli automatique sur Prophet).
