# ORBIT

**ORBIT** is an Adaptive Personal Workflow Agent. It turns a user goal into an
explicit, observable workflow: understand → plan → select action → execute tool
→ observe → validate → replan if necessary → complete.

Core principle: **the LLM proposes, deterministic application code disposes.**
The controller owns all state transitions. The LLM never mutates application
state, never executes code, and never bypasses schemas.

## Status

**Phase 6 complete: full loop integration, synthesis, persistence, API.**

Phase 1 (approved): configuration, Pydantic models, append-only events, tool
contracts + registry, `LLMProvider` protocol + `MockProvider`, state machine,
controller skeleton, in-memory store.

Phase 2 (approved): `OpenRouterProvider` with structured JSON output and one
repair attempt, safe calculator, workspace-scoped file tools, explicit
unavailable web search, tool decision validation, default registry.

Phase 3 (approved): `GoalAnalyzer` (structured constraints/preferences),
`Planner` with deterministic `validate_plan` and bounded repair.

Phase 4 (approved): `Executor` (one bounded pass, typed failures),
`Observer` (normalized observations, bounded raw), execution events, workflow
ends at `AWAITING_VALIDATION` - never directly at `COMPLETED`.

Phase 5 (approved): `Validator` (Python rechecks every numeric rule against
observation evidence), `Replanner` (targeted proposals, identical-plan
rejection, bounded repair), controller replan/validation lifecycle,
`AdaptiveLoop` (`execute → validate → replan`, bounded by `MAX_REPLANS`),
plan-revision history, atomic `replace_plan`, stale-observation guards,
app-wide process-launch/FFI ban.

Phase 6 additions:
- `Synthesizer`: final answer composition **after PASS only** - constraint
  claims are cross-checked against the validation's satisfied set (LLM-side
  post-check + controller-side `record_synthesis` gate); no tools, no state
- Clarification loop: analysis may demand required inputs
  (`AWAITING_CLARIFICATION`), answers are recorded, then planning proceeds
  with the answers as binding prompt context
- `WorkflowOrchestrator`: single state-dispatch entry (`run(goal)`) driving
  analyze → clarify → plan → execute/validate/replan → synthesize → complete,
  with structured failure reports (`analysis_error`, `planning_error`,
  `infeasible`, `awaiting_clarification`, `already_terminal`, ...)
- `PersistentStateStore`: SQLite persistence for workflows + append-only
  event logs (restart recovery at `READY` / `AWAITING_VALIDATION`;
  `recover_interrupted_tasks` resets crashed `RUNNING` tasks)
- FastAPI: `POST /workflows`, `GET /workflows/{id}`,
  `POST /workflows/{id}/run`, `GET /workflows/{id}/events?after=` with
  404/409 error mapping and leak-proof response schemas
- `app/demo.py`: fully offline scripted demo (budget goal → 9000 FAIL →
  corrective replan → 7500 PASS → synthesis) used when no API key is set
- 3 new events: `clarification.requested`, `clarification.answered`,
  `synthesis.completed`

Not implemented (out of scope): frontend, vector DB, long-term memory,
multi-agent, background workers, arbitrary code execution, shell execution,
auth, cloud/distributed infrastructure.

## Setup

```bash
python -m pip install -e ".[dev]"
```

## Run tests

```bash
python -m pytest
```

588 tests, all fully offline: no credentials, no internet, no external
services. The OpenRouter tests use `httpx.MockTransport` (never a real
request).

## Run the API

```bash
uvicorn --factory app.api.app:create_app
```

Without `OPENROUTER_API_KEY` the app serves the offline scripted demo
provider; with a key it routes to OpenRouter. Endpoints:

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/workflows` | create a workflow from `{"goal": "..."}` → 201 |
| `GET` | `/workflows/{id}` | structured status (no raw observations/prompts) |
| `POST` | `/workflows/{id}/run` | drive the workflow; optional `{"answers": {...}}` |
| `GET` | `/workflows/{id}/events?after=N` | append-only decision log page |

Errors: unknown workflow → `404`, controller rejection (e.g. missing
clarification answer) → `409`, malformed input → `422`.

## Configuration

Copy `.env.example` and set variables as needed. No API key is required to
run tests.

| Variable | Default | Purpose |
|---|---|---|
| `OPENROUTER_API_KEY` | *(none)* | OpenRouter key (never logged) |
| `OPENROUTER_MODEL` | `nvidia/nemotron-3-ultra` | Runtime LLM model id |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | OpenRouter base URL |
| `MAX_REPLANS` | `3` | Replan attempts before `FAILED` |
| `MAX_PLAN_TASKS` | `7` | Structural limit on plan size |
| `LLM_TIMEOUT_S` | `30` | Per LLM call timeout |
| `LLM_TEMPERATURE` | `0.2` | LLM temperature |
| `LLM_MAX_RETRIES` | `2` | Bounded LLM retries |
| `LLM_BACKOFF_S` | `0.5` | LLM retry backoff base |
| `TOOL_TIMEOUT_S` | `10` | Default tool timeout |
| `ORBIT_WORKSPACE` | `workspace` | Root dir for file tools |
| `MAX_FILE_BYTES` | `1000000` | File tool size limit |
| `WEB_SEARCH_PROVIDER` | `none` | Only `none` in Phase 2 |
| `WEB_SEARCH_API_KEY` | *(none)* | Future search provider key |
| `WEB_SEARCH_ENDPOINT` | *(none)* | Future search endpoint |

## Security notes

- **Calculator**: hand-written tokenizer + recursive-descent parser; accepts
  only numbers and `+ - * / ( )`. No Python evaluation of any kind.
- **File tools**: reject absolute paths, `..` components, null bytes, and any
  path that resolves outside `ORBIT_WORKSPACE` (symlinks included).
- **OpenRouter**: the API key is sent only in the `Authorization` header;
  it is redacted from `repr()`, and every outbound error message is
  sanitized so the key can never appear in logs or events.
- **LLM output**: structured JSON only, validated against Pydantic schemas;
  one repair attempt, then a typed error. Model output never executes code.
- **Synthesis**: runs only after a Python-verified `PASS`; claims outside the
  validation's satisfied constraints are refused (LLM-side and
  controller-side); no chain-of-thought is stored or returned.
- **API**: responses carry structured status, digests, and decision
  summaries only - never raw observation payloads, prompts, keys, or
  reasoning traces.

## Layout

```
app/
├── main.py             # entry point (skeleton)
├── config/settings.py  # env-driven configuration
├── agent/              # state machine, controller, goal analyzer, planner,
│                       # executor, observer, validator, replanner,
│                       # adaptive loop, synthesizer, workflow orchestrator
├── llm/                # provider protocol, schemas, mock (+ routed mock),
│                       # OpenRouter
├── tools/              # contracts, registry, calculator, file tools,
│                       # web search, decision validation, execution
├── state/              # models, append-only events, in-memory store,
│                       # SQLite persistent store
├── api/                # FastAPI app + request/response schemas
├── demo.py             # offline scripted demo provider
tests/                  # offline unit + integration + architecture tests
```

## Runtime model

ORBIT runs on **NVIDIA Nemotron 3 Ultra through OpenRouter** behind the
`LLMProvider` abstraction, swappable for other OpenRouter models or local
models without changing agent code.
