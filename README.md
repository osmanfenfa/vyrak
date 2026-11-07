<div align="center">

# Zelvryn

**Catch a hijacked account or a misbehaving AI agent before it walks off with your customers' data.**

Post-login behavioural monitoring for SaaS: humans, AI agents and service accounts.

</div>

---

## What it does

Login checks who you are once. Zelvryn watches what every identity does *after* login and scores each action against that identity's own normal behaviour, against similar identities, and against hard limits on data volume. Each action gets a decision with a plain-language reason:

```
support-agent-3 (AI agent)  read → read → write → read        ALLOW
                            export 4,000 records                CHALLENGE
  "4,000 records exported in the current 10-minute window exceeds the
   new-identity limit of 1,000."
```

It works from **server-side events only** (action type, resource, timestamp, record count). There's no browser SDK, so it covers API traffic and AI agents, which have no mouse or keyboard for behavioural-biometrics products to measure.

## How detection works

Every signal is label-free, so it works from day one without attack data:

| Signal | Catches | Learns from |
|---|---|---|
| **Volume baseline** (records per 10-min window) | bulk export, mass delete, slow exfiltration | this identity's clean history |
| **Sequence model** (Markov chain over actions) | rogue agents, hijacked sessions doing unusual things | this identity's clean history |
| **Peer group** (compared with similar identities) | new identities behaving unlike their role | the tenant's other identities |
| **Machine timing** (human identity, near-zero timing variation) | scripts or undeclared bots driving a human account | — |
| **New-identity sensitive burst** | brand-new identity going straight for sensitive data | — |
| **First privileged action** (first admin/config by an established identity) | privilege escalation | this identity's clean history |
| Action-mix similarity, randomness, too-fast timing | general deviation | this identity's clean history |
| *Optional:* calibrated gradient-boosting classifier | scripted behaviour (human identities only) | **analyst-labelled alerts** |

Signals lower a 0–100 trust score: above 50 means ALLOW, 50 or below CHALLENGE, 20 or below BLOCK.

**Protections against attackers training the system:** an action only updates the baseline if the session is in good standing and no severe signal fired. An identity's kind (human, agent or service) is fixed after its first session, so a caller can't relabel a human as an "agent" to dodge the automation checks.

## Evaluation (synthetic; read the caveat)

> **These numbers come from synthetic identities and red-team scripts in [simulation/](simulation/), not real customer traffic.** They show the engine behaves as designed. Real-world performance must be measured in shadow mode with a design partner.

`python scripts/evaluate_detection.py` (seed 42, 20 victims per scenario, ML off):

| Attack scenario | Detected | Median actions to catch | Median records leaked first |
|---|---|---|---|
| Stolen token, bulk export | 100% | 6.5 | 0 |
| Low-and-slow exfiltration | 100% | 6 | 349 |
| Rogue / prompt-injected agent | 100% | 3.5 | 9 |
| Script driving a human account | 100% | 20 | 0 |
| Privilege escalation | 100% | 5 | 14 |
| New identity smash-and-grab | 100% | 2.5 | 264 |

Normal traffic: **2.4%** of clean sessions from established identities and **0.8%** of new-identity sessions reached CHALLENGE, with about 3.6 anomalous actions per 1,000. CI regenerates this report on every push.

## Accounts and onboarding

There is no public sign-up. Access is invite-only:

1. **You (the operator)** get a `platform_admin` account, created at startup from `BOOTSTRAP_ADMIN_EMAIL` and `BOOTSTRAP_ADMIN_PASSWORD`.
2. In the dashboard's **Customers** tab you onboard a customer: you enter the organisation name and its first admin's email, and get a one-time **invite link** to send them. New customers start in shadow mode.
3. Their admin opens the link, sets a password, and invites analysts and viewers from the **Team** tab. They can also create API keys for their backend under **Integrations**.
4. Everyone signs in with email and password.

Security details:
- Passwords are hashed with scrypt.
- Five failed sign-ins lock an account for 15 minutes.
- Invite and reset links work once and expire (7 days and 2 hours).
- Disabling someone, changing their role, or a password change or reset signs that person out everywhere at once.
- Suspending a customer cuts off all its people and API keys immediately.

Email delivery isn't connected yet, so invite and reset links are shown on screen for the admin to copy and send.

## Who can do what

Every request runs as a **principal**: a subject in a **tenant** (customer organisation) holding **roles**. All data (profiles, sessions, alerts, audit) is isolated per tenant.

| Role | Can |
|---|---|
| `platform_admin` | You, the operator: onboard, suspend and reactivate customers; global thresholds; model registry |
| `admin` | Customer admin: invites and manages their team, API keys, shadow/enforce setting, profile resets, plus everything an analyst can do |
| `analyst` | Reads and labels alerts, reads sessions, profiles and audit |
| `viewer` | Read-only access to alerts, summaries and profile statistics |
| `integration` | API-key role for the customer's backend. Starts sessions and sends actions/events for any identity in its tenant |
| `user` | An end user in direct mode: own sessions and own profile (GDPR delete) |

Credentials accepted as `Authorization: Bearer …`:

- **Session token** from `POST /auth/login` (email and password). It's re-checked against the account on every request.
- **API key** `zk_<id>_<secret>`. Stored hashed and revocable.
- **JWT from your own identity provider** (claims: `sub`, `exp`, `tenant`, `roles`), for SSO setups.

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env    # set JWT_SECRET_KEY, BOOTSTRAP_ADMIN_EMAIL, BOOTSTRAP_ADMIN_PASSWORD
uvicorn api.main:app --reload --port 8000
cd dashboard && npm install && npm run dev        # http://localhost:5173
```

Sign in with your bootstrap email and password, onboard a customer in **Customers**, and open the invite link in a private window to act as that customer's admin.

Once a customer's admin has created an API key, their backend forwards events (simplest integration, no SDK):

```bash
curl -s localhost:8000/events -H "authorization: Bearer $KEY" -H 'content-type: application/json' -d '{
  "events": [{"identity_id":"support-agent-3","identity_kind":"agent","session_id":"run-42",
              "action_type":"export","resource_id":"customers","records":4000}]}'
```

For real-time enforcement, use `POST /session/start` and then `POST /session/{id}/action` for each protected operation. Full API docs are at `/docs`.

**Rollout path:** new customers run in **shadow mode**. Every decision comes back as ALLOW, with `would_be_decision` showing what enforcement would have done, and alerts still open for analysts. The customer's admin switches to enforcement under **Integrations** when ready.

## Turning analyst labels into a better model

```bash
curl -s localhost:8000/alerts/training-export -H "authorization: Bearer $ANALYST" > labels.jsonl
python scripts/train_from_labels.py labels.jsonl      # calibrated GBM, cross-validated
# → registered as a new version; roll out with POST /admin-gate/models/canary
```

Models are loaded only if their SHA-256 matches the value recorded at registration, because loading a model file can execute code.

## Architecture

```
api/            FastAPI app: auth (roles, JWT, API keys), session service, routers
core/           engine (trust + decisions), profiles, features, classifiers
detectors/      label-free detectors: sequence, volume, peer group
profiles/       Redis profile store (tenant-scoped, per-identity locks)
coldstart/      new-identity ramp-up, population prior, device/IP risk
simulation/     synthetic identities and red-team scenarios
scripts/        evaluation, training, shadow tooling
ml/             model registry (checksums, canary), trainer, feature store
dashboard/      React console: sign-in, customers, team, alerts, integrations
```

All shared state (profiles, sessions, alerts, keys, rate limits, peer baselines) lives in **Redis**, so any number of API workers can serve any request. Each identity is locked during evaluation so concurrent requests can't lose updates. Without Redis, development falls back to in-memory state; **production refuses to start without Redis** or with a weak `JWT_SECRET_KEY`.

## Deploying

```bash
cp .env.example .env    # set JWT_SECRET_KEY (32+ random chars), REDIS_URL, CORS_ORIGINS,
                        # BOOTSTRAP_ADMIN_EMAIL/PASSWORD, DASHBOARD_URL
docker build -t zelvryn-api .
docker run --env-file .env -p 8000:8000 zelvryn-api
```

The image runs as a non-root user with a health check and several workers. Set `TRUST_PROXY_HEADERS=true` only behind a reverse proxy you control.

## Tests

```bash
pip install pytest pytest-timeout fakeredis
pytest -q          # ~300 tests: unit, API/roles/tenancy, attack scenarios
```

## Status and known limits

- **No real-world validation yet.** All accuracy figures are synthetic. The next milestone is a shadow-mode pilot with a design partner.
- The PostgreSQL layer (`db/`) and Kafka processor (`ingestion/stream_processor.py`) exist but aren't wired into the API.
- Each tenant's data is isolated, but global thresholds are shared and only per-tenant shadow mode is configurable. Per-tenant tuning is a natural next step.
- The audit log is an append-only JSONL file per instance. Ship it to a SIEM or object storage in production.
- Invite and reset links are shown to the admin rather than emailed. Connecting an email provider is the next step for self-service password reset.
- With no Redis, accounts and customers live in memory and are lost on restart. Use Redis for anything real.
