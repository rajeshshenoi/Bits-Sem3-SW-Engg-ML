# Testing in production and security (Task 9)

## 1. Testing / experimentation in production: shadow deployment → canary → A/B

Offline metrics (recall 0.816, PR-AUC 0.844, ECE 0.030 on the synthetic hold-out set) are necessary
but not sufficient. Real fleets differ from the training data, and brake faults are only confirmed
weeks later at the workshop. A new model version (the *challenger*, e.g. `brakeguard-rf-v2.1`) is
therefore rolled out in three stages against the current *champion*.

| Stage | Traffic | What users see | Exit criteria |
|---|---|---|---|
| **1. Shadow** (2-4 weeks) | 100 % of requests are mirrored to the challenger | Only champion results; challenger output is logged, never shown | No errors; p95 latency < 500 ms; challenger and champion agree on ≥ 90 % of HIGH/LOW decisions; for vehicles whose workshop outcome is already known, challenger recall ≥ champion recall |
| **2. Canary** (1-2 weeks) | 5 % → 25 % → 50 % of vehicles (sticky by `vehicle_id` hash) | Challenger results for the canary cohort | No rise in error rate or latency; HIGH-alert rate stays within ±20 % of champion; no safety escalations from fleet managers |
| **3. A/B / full rollout** | 50/50, then 100 % | Challenger | After the 30-day label horizon, challenger recall is not significantly lower (one-sided test, α = 0.05) and false-alarm rate is lower or equal; then promote, keeping the old version in the registry for one-click rollback |

**Why shadow first:** the cost of a missed brake fault is a safety incident, so a challenger must
never influence a real maintenance decision before its behaviour on live data is known. Shadow mode
gives that evidence at zero risk to vehicles.

**How it maps onto this code base**

* `ModelRegistry` already versions artefacts (`ModelBundle.version`, `trained_at`, metrics,
  training data quality). Production would hold two bundles: champion and challenger.
* `BrakeFaultPredictor` is stateless, so the `ServiceContainer` can hold a second predictor.
  The route handler scores with the champion, returns that result, and submits the challenger call
  to a background task that writes to the `AuditLog` with its own `model_version`.
* `AuditLog` records include `prediction_id`, `model_version` and `fault_probability`. Joined later
  with workshop outcomes, this gives the online recall/precision of each version.
* `/api/v1/monitoring` already reports p50/p95 latency, served-prediction counts and PSI drift.
  These are the guard-rail metrics watched during canary. A drift alert (PSI > 0.25) pauses a rollout.
* Automatic rollback: if the canary cohort's error rate or p95 latency breaches its threshold, the
  router sends all traffic back to the champion. The orchestrator (e.g. Kubernetes/Argo Rollouts)
  does this using the `/ready` probe and these metrics.

## 2. Security consideration: input validation and model access control

**Threat.** BrakeGuard influences safety-critical maintenance decisions, so an attacker or a faulty
gateway that sends crafted telemetry could (a) suppress a real HIGH alert (an evasion attack), (b)
poison the feature store and drift statistics, or (c) query the model at high volume to
reverse-engineer it. The Assignment I prototype accepted any JSON. The research notebook
shows `pad_wear_pct = 450` being scored as a valid vehicle.

**Controls implemented**

1. **Schema-driven input validation** (`brakeguard/api/schemas.py`): every field has a physical
   range taken from `TELEMETRY_SCHEMA` (e.g. `0 ≤ pad_wear_pct ≤ 100`). `extra="forbid"` rejects
   unknown fields, `vehicle_id` must match `^[A-Z0-9][A-Z0-9-]{2,31}$`, and batches are capped at 500
   items. Violations return **422** with a uniform error envelope and are logged as WARNING. This
   shrinks the space of adversarial inputs to physically plausible ones and blocks injection through
   identifiers.
2. **Authentication / model access control** (`brakeguard/api/security.py`): when
   `BRAKEGUARD_API_KEY` is set, every `/api/v1/*` route requires an `X-API-Key` header, compared in
   constant time (`hmac.compare_digest`). The key comes from the environment, never from code.
   `/health` and `/ready` stay public for orchestration probes.
3. **Safe error handling**: unexpected exceptions return a generic 500 without stack traces.
   Details go only to the server log, under a `request_id` that is also returned to the client.
4. **Bounded state**: the audit log, event log and drift window are bounded `deque`s, so request
   floods cannot exhaust memory.
5. **Drift monitoring as a detection control**: a coordinated injection of manipulated telemetry
   shows up as PSI alerts on `/api/v1/monitoring`.
6. **Hardened container**: multi-stage Docker build, non-root user, read-only filesystem in
   `docker-compose.yml`, and lint + tests run as a build gate.

**Further hardening for production (not implemented in the prototype):** mutual TLS between
vehicle gateways and the API, per-client rate limiting at the API gateway, signed model artefacts
(only load a `joblib` file whose hash matches the registry entry, because unpickling an untrusted
file executes code), and role-based access so only the MLOps pipeline can write to the model
registry.
