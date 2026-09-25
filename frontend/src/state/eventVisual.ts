/**
 * Pure display derivations — no state, no side effects.
 *
 * Everything here reads REAL backend events (seq, type, payload, summary)
 * and turns it into presentation data: trace categories, the plan/task
 * column, the replan transition chain, and validation/synthesis outcomes.
 * Nothing is invented — absent data stays absent.
 */

import type { EventView, EventType } from '../api/types';

export type EventCategory =
  | 'PLAN'
  | 'TOOL'
  | 'RESULT'
  | 'CALC'
  | 'VALIDATE'
  | 'NOTICE'
  | 'ADAPT'
  | 'ERROR'
  | 'COMPLETE';

const CATEGORY: Record<EventType, EventCategory> = {
  'workflow.started': 'NOTICE',
  'goal.analyzed': 'NOTICE',
  'state.changed': 'NOTICE',
  'clarification.requested': 'NOTICE',
  'clarification.answered': 'NOTICE',
  'workflow.planned': 'PLAN',
  'workflow.replanned': 'PLAN',
  'task.started': 'TOOL',
  'tool.decision_proposed': 'TOOL',
  'tool.called': 'TOOL',
  'tool.completed': 'TOOL',
  'execution.started': 'CALC',
  'execution.finished': 'RESULT',
  'task.completed': 'RESULT',
  'observation.created': 'RESULT',
  'validation.started': 'VALIDATE',
  'validation.passed': 'VALIDATE',
  'execution.failed': 'ERROR',
  'tool.failed': 'ERROR',
  'validation.failed': 'ERROR',
  'replanning.exhausted': 'ERROR',
  'replanning.started': 'ADAPT',
  'replan.proposed': 'ADAPT',
  'replan.accepted': 'ADAPT',
  'replan.rejected': 'ADAPT',
  'plan.validation_failed': 'ADAPT',
  'synthesis.completed': 'COMPLETE',
  'workflow.completed': 'COMPLETE',
};

export function eventCategory(type: EventType): EventCategory {
  return CATEGORY[type];
}

/** "2026-09-25T18:23:36.880177Z" -> "18:23:36" (string-safe, no Date parsing). */
export function clockTime(iso: string): string {
  const time = iso.slice(11, 19);
  return /^\d{2}:\d{2}:\d{2}$/.test(time) ? time : iso;
}

// ---------------------------------------------------------------------------
// Payload readers (JsonValue -> narrow, safe)
// ---------------------------------------------------------------------------

function asString(value: unknown): string | null {
  return typeof value === 'string' ? value : null;
}

function asNumber(value: unknown): number | null {
  return typeof value === 'number' ? value : null;
}

function asStringArray(value: unknown): string[] | null {
  if (!Array.isArray(value)) return null;
  if (!value.every((item): item is string => typeof item === 'string')) return null;
  return value;
}

function summaryAfterColon(summary: string | null): string | null {
  if (!summary) return null;
  const index = summary.indexOf(': ');
  if (index < 0 || index + 2 >= summary.length) return null;
  return summary.slice(index + 2);
}

function lastIndexOfType(events: readonly EventView[], type: EventType): number {
  for (let i = events.length - 1; i >= 0; i -= 1) {
    if (events[i]?.type === type) return i;
  }
  return -1;
}

// ---------------------------------------------------------------------------
// Plan / task column
// ---------------------------------------------------------------------------

export type TaskStatus = 'pending' | 'active' | 'done' | 'failed';

export interface PlanTask {
  id: string;
  title: string | null;
  status: TaskStatus;
}

export interface PlanInfo {
  version: number | null;
  tasks: PlanTask[];
}

/**
 * Replays the event log to recover the CURRENT plan: the latest
 * workflow.planned / replan.proposed defines the task ids and version,
 * task.started/completed/failure events move each task through its lifecycle,
 * and replan.accepted's preserved_completed keeps finished work marked done
 * across plan revisions (the backend does not re-execute preserved tasks).
 */
export function derivePlan(events: readonly EventView[]): PlanInfo {
  let version: number | null = null;
  let tasks: PlanTask[] = [];
  const titles = new Map<string, string>();
  let preserved = new Set<string>();

  const resetTasks = (ids: string[]): void => {
    tasks = ids.map((id) => ({
      id,
      title: titles.get(id) ?? null,
      status: preserved.has(id) ? 'done' : 'pending',
    }));
  };

  const markPreserved = (ids: string[]): void => {
    for (const id of ids) {
      const index = tasks.findIndex((task) => task.id === id);
      const task = index >= 0 ? tasks[index] : undefined;
      if (task) tasks[index] = { ...task, status: 'done' };
    }
  };

  for (const event of events) {
    if (event.type === 'workflow.planned' || event.type === 'replan.proposed') {
      const ids = asStringArray(event.payload['task_ids']);
      if (ids) resetTasks(ids);
      const nextVersion = asNumber(event.payload['version']);
      if (nextVersion !== null) version = nextVersion;
      continue;
    }

    if (event.type === 'replan.accepted') {
      const ids = asStringArray(event.payload['preserved_completed']);
      if (ids) {
        preserved = new Set(ids);
        markPreserved(ids);
      }
      continue;
    }

    if (!event.task_id) continue;

    if (event.type === 'task.started') {
      const title = summaryAfterColon(event.decision_summary);
      if (title) titles.set(event.task_id, title);
      const index = tasks.findIndex((task) => task.id === event.task_id);
      const task = index >= 0 ? tasks[index] : undefined;
      if (task) tasks[index] = { ...task, status: 'active', title: title ?? task.title };
      continue;
    }

    const index = tasks.findIndex((task) => task.id === event.task_id);
    const task = index >= 0 ? tasks[index] : undefined;
    if (!task) continue;

    if (event.type === 'task.completed') {
      tasks[index] = { ...task, status: 'done' };
    } else if (event.type === 'execution.failed' || event.type === 'tool.failed') {
      tasks[index] = { ...task, status: 'failed' };
    }
  }

  return { version, tasks };
}

// ---------------------------------------------------------------------------
// Replan transition chain
// ---------------------------------------------------------------------------

export interface ChainStep {
  label: string;
  detail: string | null;
  state: 'done' | 'current' | 'pending';
}

/**
 * The adaptation story: constraint failed → validation → adapt → new plan →
 * execution resumes. Steps whose source events have not happened yet render
 * as pending, so the chain also reads correctly mid-replan.
 */
export function deriveReplanChain(events: readonly EventView[]): ChainStep[] | null {
  const failIndex = lastIndexOfType(events, 'validation.failed');
  const startIndex = lastIndexOfType(events, 'replanning.started');
  if (failIndex < 0 && startIndex < 0) return null;

  const chainStart = Math.max(failIndex, startIndex);
  const fail = failIndex >= 0 ? events[failIndex] : undefined;
  const start = startIndex >= 0 ? events[startIndex] : undefined;
  const acceptIndex = lastIndexOfType(events, 'replan.accepted');
  const accepted = acceptIndex >= 0 ? events[acceptIndex] : undefined;
  const proposeIndex = lastIndexOfType(events, 'replan.proposed');
  const resumed = events.some(
    (event, index) => index > chainStart && event.type === 'execution.started',
  );

  const reason = asString(fail?.payload['reason_summary']) ?? fail?.decision_summary ?? null;
  const status = fail ? asString(fail.payload['status']) : null;
  const evidence = fail ? asNumber(fail.payload['evidence_count']) : null;
  const validationDetail =
    status !== null || evidence !== null
      ? [status && `status ${status}`, evidence !== null && `evidence ${evidence}`]
          .filter(Boolean)
          .join(' · ')
      : start?.decision_summary ?? null;

  const replanCount = start ? asNumber(start.payload['replan_count']) : null;
  const limit = start ? asNumber(start.payload['limit']) : null;
  const adaptDetail =
    replanCount !== null && limit !== null ? `replan ${replanCount + 1} of ${limit}` : null;

  const previous = accepted ? asNumber(accepted.payload['previous_version']) : null;
  const version = accepted ? asNumber(accepted.payload['version']) : null;
  const newPlanDetail =
    previous !== null && version !== null ? `v${previous} → v${version}` : null;

  const states: Array<'done' | 'current' | 'pending'> = [];
  const doneFlags = [
    failIndex >= 0,
    startIndex >= 0,
    proposeIndex >= 0 || acceptIndex >= 0,
    acceptIndex >= 0,
    resumed,
  ];
  let currentFound = false;
  for (const done of doneFlags) {
    if (done) {
      states.push('done');
    } else if (!currentFound) {
      states.push('current');
      currentFound = true;
    } else {
      states.push('pending');
    }
  }

  return [
    { label: 'Constraint failed', detail: reason, state: states[0] ?? 'pending' },
    { label: 'Validation', detail: validationDetail, state: states[1] ?? 'pending' },
    { label: 'Adapt', detail: adaptDetail, state: states[2] ?? 'pending' },
    { label: 'New plan', detail: newPlanDetail, state: states[3] ?? 'pending' },
    { label: 'Execution resumes', detail: null, state: states[4] ?? 'pending' },
  ];
}

// ---------------------------------------------------------------------------
// Validation + synthesis outcomes (final result)
// ---------------------------------------------------------------------------

export interface ValidationInfo {
  status: string | null;
  reason: string | null;
  satisfied: string[];
  violated: string[];
  unresolved: string[];
  evidence: number | null;
}

/** Latest validation.started/passed/failed payload — constraint outcomes. */
export function deriveValidation(events: readonly EventView[]): ValidationInfo {
  const index = Math.max(
    lastIndexOfType(events, 'validation.passed'),
    lastIndexOfType(events, 'validation.failed'),
    lastIndexOfType(events, 'validation.started'),
  );
  const event = index >= 0 ? events[index] : undefined;
  const payload = event?.payload ?? {};
  return {
    status: asString(payload['status']),
    reason: asString(payload['reason_summary']) ?? event?.decision_summary ?? null,
    satisfied: asStringArray(payload['satisfied_constraints']) ?? [],
    violated: asStringArray(payload['violated_constraints']) ?? [],
    unresolved: asStringArray(payload['unresolved_constraints']) ?? [],
    evidence: asNumber(payload['evidence_count']),
  };
}

export interface SynthesisInfo {
  keyResults: number | null;
  caveats: number | null;
  nextActions: number | null;
}

/** Latest synthesis.completed payload — result counters. Null when absent. */
export function deriveSynthesis(events: readonly EventView[]): SynthesisInfo | null {
  const index = lastIndexOfType(events, 'synthesis.completed');
  if (index < 0) return null;
  const payload = events[index]?.payload ?? {};
  return {
    keyResults: asNumber(payload['key_results']),
    caveats: asNumber(payload['caveats']),
    nextActions: asNumber(payload['next_actions']),
  };
}
