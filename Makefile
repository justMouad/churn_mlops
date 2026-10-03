.PHONY: install train train-quick test lint run drift docker-build docker-run
install:
	pip install -r requirements.txt -r requirements-lstm.txt -r requirements-dev.txt
train:
	python scripts/train_all.py
train-quick:
	python scripts/train_all.py --quick
test:
	pytest -q --cov=churnops --cov=app
lint:
	ruff check .
run:
	uvicorn app.main:app --reload --port 8000
drift:
	python scripts/check_drift.py --simulate-drift
docker-build:
	docker build -t churn-forecast-api:local .
docker-run:
	docker run --rm -p 8000:8000 churn-forecast-api:local
