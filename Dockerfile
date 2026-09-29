# ClimaGuard — Flask dashboard container.
#
# The image needs the DVC outputs, so pull them before building:
#   dvc pull
#   docker build -t climaguard .
#   docker run --rm -p 5000:5000 climaguard
# Then open http://localhost:5000

FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOST=0.0.0.0 \
    PORT=5000

WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000
CMD ["python", "app.py"]
