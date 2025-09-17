# Learning Recommender Platform

An AI-assisted learning recommender for tech professionals, built with platform engineering practices: FastAPI backend, Streamlit frontend, Snowflake-backed persistence, CI/CD, containerization, observability, and Kubernetes manifests. Includes lightweight MLOps (model versioning, tracking, deployment helpers) and an optional real-time prediction service scaffold.

## Features

- **Personalized Learning Plans**: AI and rule-based plans using transformers and a skill dependency graph
- **REST API**: FastAPI endpoints for users, profiles, learning paths, and skill graph queries
- **Database Integration**: SQLAlchemy ORM targeting Snowflake
- **Frontend**: Streamlit UI for profile creation and recommendations
- **Observability (lightweight)**: Prometheus `/metrics`, basic request counters and latency histograms
- **Containerization**: Dockerfile + docker-compose for local dev
- **CI/CD (lightweight)**: GitHub Actions for lint, type-check, and tests
- **Kubernetes Manifests**: Minimal Deployment and Service for backend
 - **MLOps**: MLflow/ClearML tracking, local model registry, deployment helper, metrics

## Project Structure

```
backend/                 # FastAPI backend
frontend/                # Streamlit app
models/                  # AI recommender, skill graph, DB models
services/                # Database service and optional real-time service
config/                  # Database configuration (Snowflake)
k8s/                     # Minimal Kubernetes manifests
.github/workflows/       # CI pipeline
Dockerfile
docker-compose.yml
requirements.txt
```

## Architecture

- **Backend (FastAPI)**: Exposes REST endpoints for users, profiles, learning paths, skills graph, health, and metrics.
- **Models**:
  - `models/ai_recommender.py`: HuggingFace transformers-backed recommender with rule-based fallback.
  - `models/skill_graph.py`: Graph-based dependencies for generating learning paths.
  - `models/database_models.py`: SQLAlchemy ORM models.
- **Data layer**:
  - `services/database_service.py`: DB operations; uses `config/database.py` (Snowflake).
- **MLOps**:
  - `mlops/model_manager.py`: Register, track, deploy, and monitor models (MLflow, ClearML, Prometheus).
- **Optional real-time service**: `services/real_time_prediction.py` async service with cache and Kafka-ready scaffold.
- **Frontend**: `frontend/app.py` (Streamlit).
- **Observability**: `/metrics` endpoint (Prometheus exposition format).

## Prerequisites

- Python 3.11+
- Docker (for containerized run)
- Optional: kubectl (for Kubernetes), MLflow/ClearML servers if you plan to use them

## Setup & Configuration

- Snowflake credentials are configured via `config/database.py`. Set the password via env var.

PowerShell (Windows):
```powershell
$Env:SNOWFLAKE_PASSWORD = "<your_password>"
```

Bash (Linux/macOS):
```bash
export SNOWFLAKE_PASSWORD="<your_password>"
```

Optional environment variables:
- `MLFLOW_TRACKING_URI` (or pass via code to `create_model_manager`)
- ClearML credentials (if using ClearML)

## Run Locally

1. Clone the repository:
```bash
git clone <repository-url>
cd learning-recommender
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Ensure Snowflake password env var is set as above.

## Usage

### Start services (Docker)
```bash
docker compose up --build
```

### Backend endpoints
- `/health`: service health
- `/metrics`: Prometheus metrics (counters, latency)
- `/user/create`, `/profile/create`: create user and profile
- `/recommend/path`: generate learning path
- `/skills/tech-taxonomy`, `/skills/path`: skills endpoints

Example requests (PowerShell):
```powershell
Invoke-RestMethod -Method GET http://localhost:8000/health

$user = @{ username="alex"; email="alex@example.com"; experience_level="Intermediate"; target_role="ml_engineer" } | ConvertTo-Json
Invoke-RestMethod -Method POST http://localhost:8000/user/create -ContentType "application/json" -Body $user

$profile = @{ current_skills=@("python","sql"); learning_goals=@("mlops") } | ConvertTo-Json
Invoke-RestMethod -Method POST "http://localhost:8000/profile/create?user_id=1" -ContentType "application/json" -Body $profile

$recBody = @{ current_skills=@("python","sql") } | ConvertTo-Json
Invoke-RestMethod -Method POST "http://localhost:8000/recommend/path?user_id=1" -ContentType "application/json" -Body $recBody
```

### Local dev without Docker

Run backend:
```bash
uvicorn backend.main:app --reload --port 8000
```

Run frontend:
```bash
streamlit run frontend/app.py --server.port 8501
```

## CI/CD

- GitHub Actions workflow `.github/workflows/ci.yml` runs lint, type checks, and tests on PRs and pushes.

Pipeline steps:
- flake8 (linting), mypy (type-checks), pytest (unit/integration)

## MLOps Usage

`mlops/model_manager.py` provides a `ModelManager` abstraction for model registration, tracking, deployment helper, and monitoring.

Minimal example:
```python
from mlops.model_manager import create_model_manager, DeploymentEnvironment
from sklearn.linear_model import LogisticRegression

mm = create_model_manager({"mlflow_uri": "http://localhost:5000"})
model = LogisticRegression().fit(X, y)
model_id = mm.register_model(
    model,
    name="rec-scorer",
    model_type="sklearn",
    training_data_hash="hash123",
    accuracy_metrics={"accuracy": 0.91, "f1_score": 0.88},
    hyperparameters={"C": 1.0}
)
mm.deploy_model(model_id, environment=DeploymentEnvironment.DEVELOPMENT)
perf = mm.monitor_model_performance(model_id)
```

Registry layout is stored under `models/registry/<name>/<version>/` with metadata JSON. MLflow and ClearML logging are invoked when available.

## Real-time Prediction Service (optional)

An async microservice scaffold exists at `services/real_time_prediction.py`:
- High-level features: request queue, Redis cache (optional), simple load balancer, Prometheus metrics, Kafka-ready consumer/producer.

Run locally:
```bash
uvicorn services.real_time_prediction:app --port 8002
```

Predict example (PowerShell):
```powershell
$body = @{ model_id="rec-scorer"; input_data=@{ text="learn mlops" }; priority="normal" } | ConvertTo-Json
Invoke-RestMethod -Method POST http://localhost:8002/predict -ContentType "application/json" -Body $body
```

## Kubernetes (optional)

- Deploy backend with manifests in `k8s/`:
```bash
kubectl apply -f k8s/backend-deployment.yaml
kubectl apply -f k8s/backend-service.yaml
```

Default service exposes the backend on port 80 (target 8000) inside the cluster. Add an Ingress or NodePort/LoadBalancer for external access.

## Monitoring & Observability

- Prometheus exposition at `/metrics` (backend). Metrics include:
  - `lr_requests_total{endpoint,method,http_status}`
  - `lr_request_latency_seconds{endpoint}`
- For the real-time service, additional metrics are exposed (queue size, cache ratio, latency) if you run that service.

## Interview Talking Points (mapped to JD)

- **CI/CD and testing**: GitHub Actions + pytest, flake8, mypy
- **Docker & Kubernetes**: Dockerfile, docker-compose, K8s Deployment/Service
- **REST & microservices**: FastAPI backend with clean endpoints; optional real-time service in `services/`
- **Monitoring**: Prometheus `/metrics` with request counters and latency histograms
- **SQL**: SQLAlchemy models and Snowflake integration in `config/database.py`
- **NLP/ML**: Transformers-backed recommender with a skill dependency graph

## Testing

- Run tests locally:
```bash
pytest -q
```

## Notes

- Secrets (e.g., Snowflake password) should be provided via environment variables.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with FastAPI, Streamlit, SQLAlchemy, HuggingFace Transformers.
