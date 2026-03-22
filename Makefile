# Makefile — Zelvryn developer workflows
# Usage: make <target>
# Run `make help` to list all targets.

.PHONY: help install dev build test lint migrate \
        generate-data train shadow simulate benchmark \
        rebuild-prior docker-up docker-down clean

PYTHON := python3
PIP    := pip3
API_URL := http://localhost:8000

# ── Help ──────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  Zelvryn — Adaptive Behavioral Signature Authentication"
	@echo "  ─────────────────────────────────────────────────────"
	@echo "  Setup"
	@echo "    make install          Install Python dependencies"
	@echo "    make generate-data    Generate synthetic training data"
	@echo "    make train            Train initial Random Forest model"
	@echo ""
	@echo "  Development"
	@echo "    make dev              Start API server (uvicorn --reload)"
	@echo "    make migrate          Run Alembic database migrations"
	@echo "    make shadow           Run shadow mode on synthetic events"
	@echo "    make simulate         Run 3-phase session simulation"
	@echo "    make benchmark        Run in-process latency benchmark"
	@echo ""
	@echo "  Quality"
	@echo "    make test             Run full test suite"
	@echo "    make test-unit        Run unit tests only"
	@echo "    make test-integration Run integration tests only"
	@echo "    make test-scenarios   Run scenario tests only"
	@echo "    make lint             Run pylint / flake8"
	@echo ""
	@echo "  Data & Models"
	@echo "    make rebuild-prior    Rebuild population prior from Redis"
	@echo "    make ingest-cicids    Ingest CICIDS2018 corpus"
	@echo ""
	@echo "  Docker"
	@echo "    make docker-up        Start full stack (postgres + redis + api)"
	@echo "    make docker-down      Stop and remove containers"
	@echo ""
	@echo "  Misc"
	@echo "    make clean            Remove __pycache__, .pyc, build artefacts"
	@echo ""

# ── Setup ─────────────────────────────────────────────────────────────────────
install:
	$(PIP) install -r requirements.txt

generate-data:
	$(PYTHON) data/synthetic/generate_synthetic.py --samples 600 --verbose

train: generate-data
	$(PYTHON) scripts/train_initial_model.py --verbose

# ── Development ───────────────────────────────────────────────────────────────
dev:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload --log-level info

migrate:
	alembic -c db/migrations/alembic.ini upgrade head

migrate-down:
	alembic -c db/migrations/alembic.ini downgrade -1

migrate-status:
	alembic -c db/migrations/alembic.ini current

shadow:
	$(PYTHON) scripts/run_shadow_mode.py --synthetic --n-users 20 --n-actions 30 --export

simulate:
	$(PYTHON) scripts/simulate_sessions.py --n-actions 20 --verbose

benchmark:
	$(PYTHON) scripts/benchmark.py --n-actions 200 --n-sessions 3

rebuild-prior:
	$(PYTHON) scripts/build_population_prior.py

ingest-cicids:
	$(PYTHON) scripts/ingest_cicids.py --path data/raw/cicids2018/

# ── Tests ─────────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v --tb=short -q

test-unit:
	pytest tests/unit/ -v --tb=short

test-integration:
	pytest tests/integration/ -v --tb=short

test-scenarios:
	pytest tests/scenarios/ -v --tb=short

test-coverage:
	pytest tests/ --cov=core --cov=coldstart --cov=ingestion --cov=ml --cov=profiles \
	       --cov-report=term-missing --cov-report=html:htmlcov -q

lint:
	@$(PYTHON) -m flake8 core/ coldstart/ ingestion/ ml/ profiles/ api/ \
	           --max-line-length=100 --ignore=E501,W503 || true

# ── Docker ────────────────────────────────────────────────────────────────────
docker-up:
	docker compose up --build -d postgres redis api
	@echo "API available at http://localhost:8000"
	@echo "Docs at http://localhost:8000/docs"

docker-up-full:
	docker compose --profile dashboard up --build -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f api

docker-migrate:
	docker compose run --rm api \
	  alembic -c db/migrations/alembic.ini upgrade head

# ── Logs & monitoring ─────────────────────────────────────────────────────────
health:
	@curl -sf $(API_URL)/health | python3 -m json.tool || echo "API not reachable"

tail-audit:
	@tail -f logs/audit.jsonl 2>/dev/null | while read line; do \
	  echo "$$line" | python3 -m json.tool; done

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/ .coverage dist/ build/ 2>/dev/null || true
	@echo "Clean complete."