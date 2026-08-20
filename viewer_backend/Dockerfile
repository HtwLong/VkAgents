FROM python:3.13-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY ontology_data ./ontology_data
COPY runs ./runs
ENV VIEWER_RUNS_DIR=/app/runs \
    VIEWER_ONTOLOGY_DIR=/app/ontology_data

EXPOSE 8000
CMD ["sh", "-c", "exec uvicorn viewer_backend.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
