# Weight Prediction ML Service

A production-grade ML service that predicts weekly and monthly weight trends using historical data from Google Sheets.

Built as a portfolio project to demonstrate MLOps practices: modular code, experiment tracking, containerization, CI/CD, and Kubernetes deployment on Azure.

## Architecture

```
Google Sheets → FastAPI (ML models) → Streamlit Dashboard
                     ↓
                  MLflow (experiment tracking)
                  Prometheus + Grafana (monitoring)
                     ↓
              Docker → AKS (Azure Kubernetes Service)
                     ↑
              GitHub Actions (CI/CD)
```

## Tech Stack

| Layer | Technology |
|---|---|
| ML | scikit-learn, XGBoost |
| API | FastAPI, uvicorn |
| Tracking | MLflow |
| Dashboard | Streamlit, Altair |
| Monitoring | Prometheus, Grafana |
| Packaging | Python 3.12, uv |
| Containers | Docker, docker-compose |
| Orchestration | Kubernetes (AKS) |
| CI/CD | GitHub Actions |

## Models

Three models trained on weekly and monthly aggregated weight data:
- **Linear Regression** — baseline
- **Gradient Boosting** (sklearn)
- **XGBoost**

Features: `last_day_weight`, `weight_change`, `avg_weight`, `rolling_mean`. Models are retrained on every API startup from live Google Sheets data.

## Run locally

```bash
# Install dependencies
uv sync

# Run all services
docker-compose up --build
```

| Service | URL |
|---|---|
| Dashboard | http://localhost:8501 |
| API | http://localhost:8000 |
| API docs | http://localhost:8000/docs |
| MLflow UI | http://localhost:5000 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |

## API Endpoints

```
GET /health              — liveness check
GET /predict/weekly      — next week prediction (all 3 models)
GET /predict/monthly     — next month prediction (all 3 models)
GET /metrics/models      — latest MLflow metrics
GET /metrics             — Prometheus metrics
```

## CI/CD

GitHub Actions pipeline on every push to `main`:
1. **test** — ruff lint + pytest
2. **build** — Docker images pushed to GitHub Container Registry
3. **deploy** — manual trigger

## Kubernetes (AKS)

```bash
kubectl apply -f k8s/api-deployment.yaml
kubectl apply -f k8s/dashboard-deployment.yaml
```

Deployments include liveness/readiness probes and resource limits.