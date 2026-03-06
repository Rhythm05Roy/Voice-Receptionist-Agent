# syntax=docker/dockerfile:1
FROM python:3.11-slim AS builder

ENV POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1

WORKDIR /app
RUN pip install --no-cache-dir poetry

COPY pyproject.toml README.md ./
RUN poetry install --no-root --only main

COPY src ./src
RUN poetry install --only main

FROM python:3.11-slim

# Security: run as non-root user
RUN useradd --create-home appuser

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY --from=builder /usr/local /usr/local
COPY --from=builder /app /app

USER appuser

EXPOSE 8000
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
