# ---- build stage: install deps, run the quality gates, train the model ----
FROM python:3.11-slim AS build
WORKDIR /app
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt
COPY . .
# Fail the image build if lint or tests fail (CI quality gate inside the build).
RUN black --check brakeguard tests scripts app.py \
 && isort --check-only brakeguard tests scripts app.py \
 && flake8 brakeguard tests scripts app.py \
 && pytest -q -p no:cacheprovider
RUN python -m scripts.train --metrics-out artifacts/training_metrics.json

# ---- runtime stage: only runtime deps, non-root user ----
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && useradd --create-home brakeguard
COPY --from=build /app/brakeguard ./brakeguard
COPY --from=build /app/app.py ./app.py
COPY --from=build /app/artifacts ./artifacts
USER brakeguard
ENV BRAKEGUARD_LOG_LEVEL=INFO
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/ready')"
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
