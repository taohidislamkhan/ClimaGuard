# ClimaGuard — Streamlit dashboard container
#
#   docker build -t climaguard .
#   docker run --rm -p 8501:8501 climaguard
#
# Then open http://localhost:8501

FROM python:3.11-slim

# Avoid .pyc files and force stdout flushing (so Streamlit logs are live).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install CPU-only deps first to take advantage of Docker layer caching.
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Copy the rest of the source.
COPY . .

# Streamlit defaults: listen on all interfaces inside the container, default port.
EXPOSE 8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# The dashboard needs models/phase6/* and data/processed/* in the image.
# If a model file is missing, the app shows a friendly fallback (see app/advisory_app.py).
CMD ["streamlit", "run", "app/advisory_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
