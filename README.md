<div align="center">

# 🛰️ ORBIT

**Adaptive Personal Workflow Agent**

*Understand → Plan → Act → Observe → Validate → Replan → Complete*

![Status](https://img.shields.io/badge/status-phase%207%20complete-brightgreen)
![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)
![Tests](https://img.shields.io/badge/tests-690%20passing-brightgreen)
![Frontend](https://img.shields.io/badge/frontend-React%20%2B%20Vite-61DAFB?logo=vite&logoColor=black)
![API](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-yellow)
![Secrets](https://img.shields.io/badge/secrets-%E2%9C%94%20none%20committed-red)

</div>

---

**ORBIT** is an Adaptive Personal Workflow Agent. It turns a user goal into an
explicit, observable workflow: understand → plan → select action → execute tool
→ observe → validate → replan if necessary → complete.

Core principle: **the LLM proposes, deterministic application code disposes.**
The controller owns all state transitions. The LLM never mutates application
state, never executes code, and never bypasses schemas.

## 📑 Table of contents

- [Status](#status)
- [Setup](#setup)
- [Run tests](#run-tests)
- [Run the API](#run-the-api)
- [Configuration](#configuration)
- [Web search](#web-search)
- [Security notes](#security-notes)
- [Layout](#layout)
- [Runtime model](#runtime-model)
- [License](#license)

## Status

**Phase 7 complete: real web search. Full loop integration, synthesis,
persistence, API.**

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

Phase 7 additions:
- Real web search behind the existing `SearchProvider`/`WebSearchTool`
  architecture: LLM proposes `ToolDecision` → deterministic validation →
  `WebSearchTool` → `SearchProvider` → real HTTP (`httpx` only, no SDK,
  no new deps)
- `LangSearchProvider` (`app/tools/search_providers.py`): `POST
  https://api.langsearch.com/v1/web-search` with a Bearer token, bounded
  timeout, bounded retries with backoff, bounded response size, typed
  failures (`WebSearchTimeoutError`, `WebSearchAuthError`,
  `WebSearchResponseError`, `WebSearchProviderError`) mapped to error codes
  `SEARCH_TIMEOUT`, `SEARCH_AUTH_FAILED`, `SEARCH_BAD_RESPONSE`,
  `SEARCH_PROVIDER_ERROR`
- Provider selection via `WEB_SEARCH_PROVIDER` (`none` = unavailable tool,
  `langsearch` = real provider); without `LANGSEARCH_API_KEY` the tool stays
  explicitly unavailable - the app never fakes results
- Results are untrusted input: URLs must be http(s), fields are sanitized
  (API key redacted, control characters collapsed) and length-bounded before
  they reach observations/events; empty result sets are DATA (`[]`), not
  execution failures, so replanning still sees missing evidence

Not implemented (out of scope): vector DB, long-term memory,
multi-agent, background workers, arbitrary code execution, shell execution,
auth, cloud/distributed infrastructure.

## 🚀 Setup

```bash
python -m pip install -e ".[dev]"
```

## ✅ Run tests

```bash
python -m pytest
```

690 backend tests, all fully offline: no credentials, no internet, no
external services. The OpenRouter and LangSearch tests use
`httpx.MockTransport` (never a real request). Frontend tests:
`cd frontend && npm test`.

## ▶️ Run the API

```bash
uvicorn --factory app.api.app:create_app
```

Provider selection: `OPENROUTER_API_KEY` set → OpenRouter (real workflows,
including real web search). Nothing configured → the offline scripted demo.
Web search configured but no LLM key → explicit startup error (never a
silent scripted run). Endpoints:

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/workflows` | create a workflow from `{"goal": "..."}` → 201 |
| `GET` | `/workflows/{id}` | structured status (no raw observations/prompts) |
| `POST` | `/workflows/{id}/run` | drive the workflow; optional `{"answers": {...}}` |
| `GET` | `/workflows/{id}/events?after=N` | append-only decision log page |

Errors: unknown workflow → `404`, controller rejection (e.g. missing
clarification answer) → `409`, malformed input → `422`.

## ⚙️ Configuration

Copy `.env.example` to `.env` and set variables as needed; the project-root
`.env` is loaded at startup (real environment variables win over the file).
No API key is required to run tests - tests never read `.env`.

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
| `WEB_SEARCH_PROVIDER` | `none` | `none` (unavailable) or `langsearch` |
| `LANGSEARCH_API_KEY` | *(none)* | LangSearch API key (never logged) |
| `WEB_SEARCH_ENDPOINT` | *(none)* | Endpoint override (defaults to LangSearch's web-search endpoint) |

`WEB_SEARCH_API_KEY` is still accepted as a legacy alias for
`LANGSEARCH_API_KEY`.

## 🔎 Web search

```bash
# .env (gitignored) - real search activates only with a key
WEB_SEARCH_PROVIDER=langsearch
LANGSEARCH_API_KEY=<your-langsearch-key>
OPENROUTER_API_KEY=<your-openrouter-key>   # real workflows need a real LLM
```

Search runs inside real workflows, so it needs a real LLM as well: with
web search configured but no `OPENROUTER_API_KEY`, the API refuses to start
(explicit error) instead of silently replaying the offline scripted demo.

How it runs end to end:

1. The LLM proposes a `ToolDecision` for `web_search`
   (`{"query": ..., "max_results": ...}`) - it never calls HTTP itself.
2. Deterministic validation checks the tool exists in the registry, the name
   matches the hint, and the arguments match the declared JSON schema.
3. `WebSearchTool` calls the configured `SearchProvider`
   (`LangSearchProvider` → `POST https://api.langsearch.com/v1/web-search`,
   or `WEB_SEARCH_ENDPOINT` if set, authenticated with `Authorization:
   Bearer <key>`, body `{"query": ..., "count": ...}`), with a 6s timeout,
   one retry with backoff, and a 512 KB response cap.
4. The `ToolResult` becomes an observation: LangSearch's
   `data.webPages.value` entries are normalized to
   `{"title", "url", "snippet", "source"}` (`name`→title, `snippet` or
   bounded `text`→snippet, URL host→source) with bounded lengths
   (title 200 / url 300 / snippet 300 / source 100 chars), http(s)-only
   URLs, the API key redacted, and control characters collapsed. Results
   are untrusted evidence - never instructions.
5. Empty result set = `[]` (DATA), so validation/replanning can react to
   missing evidence instead of failing the execution.

Optional LangSearch features (`contents.text`, `freshness`,
`includeDomains`/`excludeDomains`) are deliberately not requested: snippets
are sufficient and full page text must never bulk up workflow state.

Limitations: one provider (LangSearch) behind a small protocol - adding
another is a new `SearchProvider` plus a routing entry in
`create_search_provider`. Without a key the tool reports
`web search is not configured` (an explicit DATA-level unavailability, not a
crash). Search result snippets are third-party claims: validation only
checks what the observations actually support.

## 🔒 Security notes

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
- **Web search**: the provider key is sent only in the `Authorization`
  header, redacted from every outbound error message, and never stored in
  events or workflow state; fetched results are treated as untrusted evidence
  (bounded fields, http(s)-only URLs, never executed; result links are not
  auto-followed).
- **API**: responses carry structured status, digests, and decision
  summaries only - never raw observation payloads, prompts, keys, or
  reasoning traces.
- **Secrets**: `.env` is gitignored and never tracked; only `.env.example`
  placeholders are committed.

## 🗂️ Layout

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
│                       # web search + LangSearch provider, decision
│                       # validation, execution
├── state/              # models, append-only events, in-memory store,
│                       # SQLite persistent store
├── api/                # FastAPI app + request/response schemas
├── demo.py             # offline scripted demo provider
tests/                  # offline unit + integration + architecture tests
```

## 🧠 Runtime model

ORBIT runs on **NVIDIA Nemotron 3 Ultra through OpenRouter** behind the
`LLMProvider` abstraction, swappable for other OpenRouter models or local
models without changing agent code.

## 📜 License

Released under the [MIT License](LICENSE).

```
MIT License · Copyright (c) 2026 rawrkey
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software.
```

See [LICENSE](LICENSE) for the full text.
