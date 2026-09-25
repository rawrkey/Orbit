/**
 * Runtime contract validation for backend JSON.
 *
 * TypeScript widens JSON imports (`state` becomes `string`), so compile-time
 * `satisfies` cannot verify fixtures against the API types. These parsers are
 * the real contract: they validate exact key sets, primitive types, and enum
 * membership, and return fully typed values for downstream assertions.
 *
 * Used by fixture tests only — the HTTP client itself never trusts payloads.
 */

import {
  EVENT_TYPES,
  TERMINATIONS,
  WORKFLOW_STATES,
  type CreateWorkflowResponse,
  type EventsResponse,
  type EventView,
  type FailureContext,
  type JsonValue,
  type RunWorkflowResponse,
  type WorkflowState,
  type WorkflowStatusResponse,
} from './types';

function record(value: unknown, label: string): Record<string, unknown> {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw new Error(`${label}: expected an object, got ${JSON.stringify(value)}`);
  }
  return value as Record<string, unknown>;
}

function exactKeys(value: Record<string, unknown>, keys: readonly string[], label: string): void {
  const actual = Object.keys(value).sort();
  const expected = [...keys].sort();
  if (JSON.stringify(actual) !== JSON.stringify(expected)) {
    throw new Error(
      `${label}: key set mismatch — expected [${expected.join(', ')}], got [${actual.join(', ')}]`,
    );
  }
}

function str(value: unknown, label: string): string {
  if (typeof value !== 'string') throw new Error(`${label}: expected string, got ${typeof value}`);
  return value;
}

function nullableStr(value: unknown, label: string): string | null {
  if (value === null) return null;
  return str(value, label);
}

function num(value: unknown, label: string): number {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    throw new Error(`${label}: expected number, got ${JSON.stringify(value)}`);
  }
  return value;
}

function bool(value: unknown, label: string): boolean {
  if (typeof value !== 'boolean') throw new Error(`${label}: expected boolean`);
  return value;
}

function enumOf<T extends string>(
  value: unknown,
  allowed: readonly T[],
  label: string,
): T {
  const parsed = str(value, label);
  if (!(allowed as readonly string[]).includes(parsed)) {
    throw new Error(`${label}: ${JSON.stringify(parsed)} not in [${allowed.join(', ')}]`);
  }
  return parsed as T;
}

function jsonRecord(value: unknown, label: string): Record<string, JsonValue> {
  return record(value, label) as Record<string, JsonValue>;
}

const FAILURE_KEYS = ['detail', 'error_code', 'stage', 'task_id'] as const;

function parseFailure(value: unknown, label: string): FailureContext | null {
  if (value === null) return null;
  const failure = record(value, label);
  exactKeys(failure, FAILURE_KEYS, label);
  return {
    stage: enumOf(failure.stage, WORKFLOW_STATES, `${label}.stage`),
    error_code: str(failure.error_code, `${label}.error_code`),
    detail: str(failure.detail, `${label}.detail`),
    task_id: nullableStr(failure.task_id, `${label}.task_id`),
  };
}

export function parseCreate(value: unknown): CreateWorkflowResponse {
  const created = record(value, 'create response');
  exactKeys(created, ['state', 'workflow_id'], 'create response');
  return {
    workflow_id: str(created.workflow_id, 'create.workflow_id'),
    state: enumOf(created.state, WORKFLOW_STATES, 'create.state'),
  };
}

const STATUS_KEYS = [
  'clarification_questions',
  'created_at',
  'event_count',
  'failure',
  'final_answer',
  'goal',
  'observation_count',
  'plan_version',
  'replan_count',
  'state',
  'task_count',
  'updated_at',
  'workflow_id',
] as const;

export function parseStatus(value: unknown): WorkflowStatusResponse {
  const status = record(value, 'status response');
  exactKeys(status, STATUS_KEYS, 'status response');
  const questions = status.clarification_questions;
  if (!Array.isArray(questions)) throw new Error('status.clarification_questions: expected array');
  return {
    workflow_id: str(status.workflow_id, 'status.workflow_id'),
    goal: str(status.goal, 'status.goal'),
    state: enumOf(status.state, WORKFLOW_STATES, 'status.state') as WorkflowState,
    replan_count: num(status.replan_count, 'status.replan_count'),
    created_at: str(status.created_at, 'status.created_at'),
    updated_at: str(status.updated_at, 'status.updated_at'),
    task_count: num(status.task_count, 'status.task_count'),
    observation_count: num(status.observation_count, 'status.observation_count'),
    event_count: num(status.event_count, 'status.event_count'),
    plan_version:
      status.plan_version === null ? null : num(status.plan_version, 'status.plan_version'),
    clarification_questions: questions.map((question, index) =>
      str(question, `status.clarification_questions[${index}]`),
    ),
    final_answer: nullableStr(status.final_answer, 'status.final_answer'),
    failure: parseFailure(status.failure, 'status.failure'),
  };
}

const RUN_KEYS = [
  'executions',
  'failure',
  'final_state',
  'replans',
  'success',
  'termination',
  'validations',
  'workflow_id',
] as const;

export function parseRun(value: unknown): RunWorkflowResponse {
  const report = record(value, 'run response');
  exactKeys(report, RUN_KEYS, 'run response');
  return {
    workflow_id: str(report.workflow_id, 'run.workflow_id'),
    success: bool(report.success, 'run.success'),
    final_state: enumOf(report.final_state, WORKFLOW_STATES, 'run.final_state'),
    executions: num(report.executions, 'run.executions'),
    validations: num(report.validations, 'run.validations'),
    replans: num(report.replans, 'run.replans'),
    termination: enumOf(report.termination, TERMINATIONS, 'run.termination'),
    failure: parseFailure(report.failure, 'run.failure'),
  };
}

const EVENT_KEYS = [
  'decision_summary',
  'payload',
  'seq',
  'state',
  'task_id',
  'timestamp',
  'type',
] as const;

function parseEvent(value: unknown, label: string): EventView {
  const event = record(value, label);
  exactKeys(event, EVENT_KEYS, label);
  const summary = event.decision_summary;
  if (summary !== null && typeof summary !== 'string') {
    throw new Error(`${label}.decision_summary: expected string | null`);
  }
  if (typeof event.task_id !== 'string' && event.task_id !== null) {
    throw new Error(`${label}.task_id: expected string | null`);
  }
  return {
    seq: num(event.seq, `${label}.seq`),
    timestamp: str(event.timestamp, `${label}.timestamp`),
    type: enumOf(event.type, EVENT_TYPES, `${label}.type`),
    state: enumOf(event.state, WORKFLOW_STATES, `${label}.state`),
    task_id: event.task_id as string | null,
    decision_summary: summary as string | null,
    payload: jsonRecord(event.payload, `${label}.payload`),
  };
}

export function parseEvents(value: unknown): EventsResponse {
  const page = record(value, 'events response');
  exactKeys(page, ['count', 'events', 'workflow_id'], 'events response');
  if (!Array.isArray(page.events)) throw new Error('events.events: expected array');
  return {
    workflow_id: str(page.workflow_id, 'events.workflow_id'),
    events: page.events.map((event, index) => parseEvent(event, `events[${index}]`)),
    count: num(page.count, 'events.count'),
  };
}
