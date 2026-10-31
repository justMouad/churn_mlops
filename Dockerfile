# syntax=docker/dockerfile:1
# ---------- builder : dépendances dans un venv ----------
FROM python:3.11-slim AS builder
ARG WITH_LSTM=true
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt requirements-lstm.txt ./
RUN pip install -r requirements.txt \
 && if [ "$WITH_LSTM" = "true" ]; then pip install torch --index-url https://download.pytorch.org/whl/cpu; fi

# ---------- runtime : image légère, utilisateur non-root ----------
FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app \
    PATH="/opt/venv/bin:$PATH" ARTIFACTS_DIR=/app/artifacts
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 1001 appuser
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY churnops ./churnops
COPY app ./app
COPY artifacts ./artifacts
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
