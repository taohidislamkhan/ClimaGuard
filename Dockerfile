# syntax=docker/dockerfile:1
# ClimaGuard — Flask dashboard container.
#
# The image needs the DVC outputs (models/, artifacts/, reports/shap, ...). Two ways:
#
#  * Local: `dvc pull` first, so they are already in the build context:
#      docker build -t climaguard .
#  * Render / CI: pass DagsHub credentials and the build pulls them itself:
#      docker build --build-arg DAGSHUB_USER=<user> --build-arg DAGSHUB_TOKEN=<token> -t climaguard .
#    (Render passes the service's environment variables as build args.) The
#    credentials only exist in the throwaway `models` stage, never in the final image.
#
#   docker run --rm -p 5000:5000 -e SECRET_KEY=<random> -v climaguard-db:/app/instance climaguard
#   docker exec -it <container> flask create-admin --email you@example.com --name You
# Then open http://localhost:5000 (SECRET_KEY is required; accounts live in the volume)

# --- stage 1: fetch the DVC outputs -------------------------------------------
FROM python:3.14-slim AS models

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /src
COPY requirements.txt ./
RUN pip install "$(grep -E '^dvc==' requirements.txt)"

COPY . .
ARG DAGSHUB_USER=""
ARG DAGSHUB_TOKEN=""
RUN if [ -n "$DAGSHUB_TOKEN" ]; then \
        dvc config core.no_scm true \
        && dvc remote modify --local origin auth basic \
        && dvc remote modify --local origin user "$DAGSHUB_USER" \
        && dvc remote modify --local origin password "$DAGSHUB_TOKEN" \
        && dvc pull \
            data/interim/clean.parquet \
            data/processed/features.parquet \
            models/best_overall_classifier.joblib \
            models/best_respiratory.joblib \
            models/best_cardio.joblib \
            models/best_vector.joblib \
            models/best_waterborne.joblib \
            models/best_heat.joblib \
            reports/shap \
            artifacts \
        && rm -rf .dvc/config.local .dvc/cache .dvc/tmp; \
    else \
        echo "No DAGSHUB_TOKEN: using the DVC outputs already in the build context."; \
    fi \
    && test -f models/best_overall_classifier.joblib && test -d artifacts \
    || (echo "Model outputs missing: run 'dvc pull' locally or set DAGSHUB_USER/DAGSHUB_TOKEN." && exit 1)

# --- stage 2: the app ---------------------------------------------------------
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOST=0.0.0.0 \
    PORT=5000

# xgboost needs the OpenMP runtime, which the slim image leaves out.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# The app never imports dvc or pytest: skip them to keep the image small.
COPY requirements.txt ./
RUN grep -vE '^(dvc|pytest)==' requirements.txt > /tmp/req.txt && pip install -r /tmp/req.txt

COPY --from=models /src /app
RUN rm -rf .dvc/tmp .dvc/cache data/raw models/candidates \
    data/processed/train.parquet data/processed/val.parquet data/processed/test.parquet

EXPOSE 5000
# One worker: in-memory rate limits + SQLite accounts (see wsgi.py).
CMD gunicorn --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:${PORT} wsgi:app
