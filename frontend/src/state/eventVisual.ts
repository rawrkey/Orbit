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
  /** Semantic tone for the step marker (colors the rail dot). */
  tone: 'fail' | 'adapt' | 'pass';
}

/**
 * The adaptive loop, read top-to-bottom: validation failed → orbit adapted →
 * plan revised → execution resumed → validation passed. Steps whose source
 * events have not happened yet render as pending, so the sequence also reads
 * correctly mid-replan. Uppercase is applied visually by CSS.
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
  const resumed = events.some(
    (event, index) => index > chainStart && event.type === 'execution.started',
  );
  const passIndex = lastIndexOfType(events, 'validation.passed');
  const passed = passIndex > chainStart ? events[passIndex] : undefined;

  const violated = fail ? asStringArray(fail.payload['violated_constraints']) ?? [] : [];
  const failReason = asString(fail?.payload['reason_summary']) ?? fail?.decision_summary ?? null;
  const failDetail = violated.length > 0 ? `${violated.join(', ')} violated` : failReason;

  const replanCount = start ? asNumber(start.payload['replan_count']) : null;
  const limit = start ? asNumber(start.payload['limit']) : null;
  const adaptDetail =
    replanCount !== null && limit !== null ? `replan ${replanCount + 1} of ${limit}` : null;

  const previous = accepted ? asNumber(accepted.payload['previous_version']) : null;
  const version = accepted ? asNumber(accepted.payload['version']) : null;
  const revisedDetail =
    previous !== null && version !== null ? `v${previous} → v${version}` : null;

  const satisfied = passed ? asStringArray(passed.payload['satisfied_constraints']) ?? [] : [];
  const passReason =
    (passed ? asString(passed.payload['reason_summary']) : null) ??
    passed?.decision_summary ??
    null;
  const passDetail = satisfied.length > 0 ? `${satisfied.join(', ')} satisfied` : passReason;

  const doneFlags = [failIndex >= 0, startIndex >= 0, acceptIndex >= 0, resumed, passed !== undefined];
  let currentFound = false;
  const states = doneFlags.map((done): 'done' | 'current' | 'pending' => {
    if (done) return 'done';
    if (!currentFound) {
      currentFound = true;
      return 'current';
    }
    return 'pending';
  });

  return [
    { label: 'Validation failed', detail: failDetail, state: states[0] ?? 'pending', tone: 'fail' },
    { label: 'Orbit adapted', detail: adaptDetail, state: states[1] ?? 'pending', tone: 'adapt' },
    { label: 'Plan revised', detail: revisedDetail, state: states[2] ?? 'pending', tone: 'adapt' },
    { label: 'Execution resumed', detail: null, state: states[3] ?? 'pending', tone: 'adapt' },
    { label: 'Validation passed', detail: passDetail, state: states[4] ?? 'pending', tone: 'pass' },
  ];
}

// ---------------------------------------------------------------------------
// Plan version history (superseded plans + what the replan changed)
// ---------------------------------------------------------------------------

export interface PlanRevision {
  version: number;
  total: number;
  /** task.completed events for this plan's task ids while it was current. */
  completed: number;
  /** Constraints violated by validation.failed events under this plan. */
  constraintFailures: number;
}

export interface AdaptLink {
  from: number;
  to: number;
  preserved: number;
  revised: number;
  added: number;
}

export interface PlanHistory {
  /** One entry per workflow.planned event, in order (length >= 2). */
  revisions: PlanRevision[];
  /** revisions[i] -> revisions[i + 1] (length = revisions.length - 1). */
  links: AdaptLink[];
}

/**
 * Replays plan versions from real events: each workflow.planned opens a
 * revision, task.completed/validation.failed inside its window count the
 * work (and constraint failures) done under it, and each replan.accepted
 * between two revisions supplies preserved_completed — from which preserved
 * / revised / new counts are derived by set comparison. Null until a second
 * plan exists. Nothing is invented: absent payloads fall back to set diffs.
 */
export function derivePlanHistory(events: readonly EventView[]): PlanHistory | null {
  const planned: Array<{ index: number; version: number; ids: string[] }> = [];
  for (let i = 0; i < events.length; i += 1) {
    const event = events[i];
    if (event?.type !== 'workflow.planned') continue;
    const version = asNumber(event.payload['version']);
    if (version === null) continue; // a plan without a version is ambiguous
    const ids = asStringArray(event.payload['task_ids']) ?? [];
    planned.push({ index: i, version, ids });
  }
  if (planned.length < 2) return null;

  const revisions: PlanRevision[] = [];
  const links: AdaptLink[] = [];

  for (let i = 0; i < planned.length; i += 1) {
    const current = planned[i];
    const next = planned[i + 1];
    if (!current) continue;
    const end = next ? next.index : events.length;

    let completed = 0;
    let constraintFailures = 0;
    for (let j = current.index + 1; j < end; j += 1) {
      const event = events[j];
      if (!event) continue;
      if (event.type === 'task.completed' && event.task_id && current.ids.includes(event.task_id)) {
        completed += 1;
      }
      if (event.type === 'validation.failed') {
        constraintFailures += asStringArray(event.payload['violated_constraints'])?.length ?? 0;
      }
    }
    revisions.push({
      version: current.version,
      total: current.ids.length,
      completed,
      constraintFailures,
    });

    if (next) {
      let preserved = 0;
      for (let j = current.index + 1; j < end; j += 1) {
        const event = events[j];
        if (event?.type !== 'replan.accepted') continue;
        preserved = asStringArray(event.payload['preserved_completed'])?.length ?? 0;
      }
      const common = current.ids.filter((id) => next.ids.includes(id)).length;
      links.push({
        from: current.version,
        to: next.version,
        preserved,
        revised: Math.max(common - preserved, 0),
        added: next.ids.filter((id) => !current.ids.includes(id)).length,
      });
    }
  }

  return { revisions, links };
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
