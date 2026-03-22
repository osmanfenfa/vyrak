<div align="center">

# Zelvryn

**Adaptive Behavioral Signature Authentication**

*Continuous session intelligence — because login is just the beginning.*

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Redis](https://img.shields.io/badge/Redis-DC382D?style=flat-square&logo=redis&logoColor=white)](https://redis.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)

</div>

---

## What is Zelvryn?

Zelvryn doesn't just check who you are at login — it watches how you behave throughout the entire session. Every action is silently scored against your personal behavioural fingerprint. If something drifts — session hijack, AI bot, insider threat — trust decays and access is blocked automatically.

```
Normal user   → read → write → search   Trust: 100  ✅ ALLOW
Stolen token  → export → admin → delete  Trust:  18  🚫 BLOCK
AI mimic bot  → [uniform timing pattern]  P(bot): 0.80  ⚠️ CHALLENGE → BLOCK
```

## How it works

Zelvryn runs two detection systems simultaneously and fuses their signals:

- **Rule engine** — cosine similarity against your profile, Shannon entropy scoring, temporal Z-score on timing
- **Random Forest classifier** — trained on a 20-dimensional behaviour vector per action window
- **Dual-signal escalation** — when both systems flag the same window, penalties compound

The classifier retrains continuously as new sessions accumulate. A GAN adversarial loop hardens it against bots that try to mimic human patterns over time.

## Quick Start

```bash
git clone https://github.com/vyrak/zelvryn.git
cd zelvryn

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env          # set DATABASE_URL and REDIS_URL
docker-compose up -d          # starts Redis + PostgreSQL
alembic upgrade head
python scripts/train_initial_model.py

make run                      # API at http://localhost:8000
```

Run the demo:

```bash
python scripts/simulate_sessions.py
```

## Core API

```
POST   /session/start              → { session_id }
POST   /session/{id}/action        → { decision, trust_score, bot_prob, anomalies }
GET    /session/{id}/summary       → full session report
DELETE /session/{id}
```

## Tech Stack

Python · FastAPI · Redis · PostgreSQL · scikit-learn · river · PyTorch · React

## Documentation

Full docs live in [`/docs`](./docs):

- [`architecture.md`](./docs/architecture.md) — system design and data flow
- [`algorithm.md`](./docs/algorithm.md) — ABSA v1 + v2 explained
- [`api_reference.md`](./docs/api_reference.md) — all endpoints with examples
- [`feature_reference.md`](./docs/feature_reference.md) — all 20 behaviour features
- [`deployment.md`](./docs/deployment.md) — Docker, Redis, environment setup
- [`dataset_guide.md`](./docs/dataset_guide.md) — CICIDS2018, CTU-13, shadow mode

## 2025 Vyrak

---

<div align="center">

*Continuous trust. Persistent intelligence. Zero compromise.*

</div>

---

## Citation

If you use Zelvryn in academic work, please cite:

```bibtex
@software{zelvryn2025,
  author    = {Vyrak},
  title     = {Zelvryn: Adaptive Behavioral Signature Authentication
               with Random Forest AI Mimic Detection},
  year      = {2025},
  url       = {https://github.com/vyrak,
  note      = {Continuous session authentication with dual-signal fusion
               and online adversarial hardening}
}
```

---

<div align="center">

**Zelvryn** — built with purpose by [Vyrak](https://github.com/vyrak)

*Continuous trust. Persistent intelligence. Zero compromise.*

HEAD
</div>
HEAD
7b49421 (conti...)
18b097e (Conti...)
