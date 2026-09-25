/**
 * Display derivations against the REAL backend fixture: plan replay, replan
 * chain, validation outcomes, categories, timestamps. No behavior is faked —
 * if these pass, the visual layers are reading genuine backend data.
 */

import { describe, expect, it } from 'vitest';

import { EVENT_TYPES, type EventView } from '../api/types';
import eventsDemo from '../api/fixtures/events_demo.json';
import {
  clockTime,
  derivePlan,
  derivePlanHistory,
  deriveReplanChain,
  deriveSynthesis,
  deriveValidation,
  eventCategory,
} from './eventVisual';

// Fixtures are contract-checked in api/fixtures.test.ts (runtime parsers).
const demo = eventsDemo.events as unknown as EventView[];

describe('eventCategory', () => {
  it('classifies every backend event type', () => {
    for (const type of EVENT_TYPES) {
      expect(eventCategory(type)).toBeTruthy();
    }
  });

  it('uses the nine presentation categories', () => {
    const seen = new Set(EVENT_TYPES.map((type) => eventCategory(type)));
    expect([...seen].sort()).toEqual([
      'ADAPT',
      'CALC',
      'COMPLETE',
      'ERROR',
      'NOTICE',
      'PLAN',
      'RESULT',
      'TOOL',
      'VALIDATE',
    ]);
  });
});

describe('clockTime', () => {
  it('extracts HH:MM:SS from backend ISO timestamps', () => {
    expect(clockTime('2026-09-25T18:23:36.880177Z')).toBe('18:23:36');
  });

  it('passes through strings it cannot parse', () => {
    expect(clockTime('not-a-time')).toBe('not-a-time');
  });
});

describe('derivePlan (real demo story)', () => {
  it('replays to the CURRENT plan: v2, both tasks completed', () => {
    const plan = derivePlan(demo);
    expect(plan.version).toBe(2);
    expect(plan.tasks).toHaveLength(2);
    expect(plan.tasks.map((task) => task.status)).toEqual(['done', 'done']);
    expect(plan.tasks[0]?.id).toBe('t1');
  });

  it('recovers task titles from execution summaries', () => {
    const plan = derivePlan(demo);
    expect(plan.tasks[0]?.title).toBe('Recompute cheaper options');
    expect(plan.tasks[1]?.title).toBe('Summarize options');
  });

  it('marks pending tasks before execution (fresh plan only)', () => {
    const fresh = derivePlan(demo.filter((event) => event.seq <= 4));
    expect(fresh.version).toBe(1);
    expect(fresh.tasks.map((task) => task.status)).toEqual(['pending', 'pending']);
  });

  it('is empty when no plan has been recorded', () => {
    expect(derivePlan([])).toEqual({ version: null, tasks: [] });
  });
});

describe('deriveReplanChain (real demo story)', () => {
  it('tells the full adaptation story in order', () => {
    const chain = deriveReplanChain(demo);
    expect(chain).not.toBeNull();
    expect(chain?.map((step) => step.label)).toEqual([
      'Validation failed',
      'Orbit adapted',
      'Plan revised',
      'Execution resumed',
      'Validation passed',
    ]);
    expect(chain?.every((step) => step.state === 'done')).toBe(true);
  });

  it('carries real details: violation, replan budget, version jump, satisfaction', () => {
    const chain = deriveReplanChain(demo);
    expect(chain?.[0]?.detail).toBe('c1 violated');
    expect(chain?.[1]?.detail).toBe('replan 1 of 3');
    expect(chain?.[2]?.detail).toBe('v1 → v2');
    expect(chain?.[3]?.detail).toBeNull();
    expect(chain?.[4]?.detail).toBe('c1 satisfied');
  });

  it('tones the steps for the rail markers', () => {
    const chain = deriveReplanChain(demo);
    expect(chain?.map((step) => step.tone)).toEqual(['fail', 'adapt', 'adapt', 'adapt', 'pass']);
  });

  it('is absent without validation/replanning events', () => {
    expect(deriveReplanChain(demo.filter((event) => event.seq < 21))).toBeNull();
  });

  it('shows later steps as pending mid-replan', () => {
    const mid = deriveReplanChain(demo.filter((event) => event.seq <= 23));
    expect(mid?.[0]?.state).toBe('done');
    expect(mid?.[1]?.state).toBe('done');
    expect(mid?.[2]?.state).toBe('current');
    expect(mid?.[4]?.state).toBe('pending');
  });
});

describe('derivePlanHistory (real demo story)', () => {
  it('replays both plan versions with real counts', () => {
    const history = derivePlanHistory(demo);
    expect(history).not.toBeNull();
    expect(history?.revisions).toEqual([
      { version: 1, total: 2, completed: 2, constraintFailures: 1 },
      { version: 2, total: 2, completed: 1, constraintFailures: 0 },
    ]);
  });

  it('derives what the replan changed from preserved_completed', () => {
    const history = derivePlanHistory(demo);
    expect(history?.links).toEqual([{ from: 1, to: 2, preserved: 1, revised: 1, added: 0 }]);
  });

  it('is absent before a second plan exists', () => {
    expect(derivePlanHistory(demo.filter((event) => event.seq <= 20))).toBeNull();
    expect(derivePlanHistory([])).toBeNull();
  });
});

describe('deriveValidation / deriveSynthesis (real demo story)', () => {
  it('reads the final PASS with its constraint outcomes', () => {
    const validation = deriveValidation(demo);
    expect(validation.status).toBe('PASS');
    expect(validation.satisfied).toEqual(['c1']);
    expect(validation.violated).toEqual([]);
    expect(validation.evidence).toBe(1);
  });

  it('reads synthesis counters', () => {
    const synthesis = deriveSynthesis(demo);
    expect(synthesis?.keyResults).toBe(3);
    expect(synthesis?.caveats).toBe(1);
    expect(synthesis?.nextActions).toBe(2);
  });

  it('stays empty on an empty log', () => {
    const validation = deriveValidation([]);
    expect(validation.status).toBeNull();
    expect(validation.satisfied).toEqual([]);
    expect(deriveSynthesis([])).toBeNull();
  });
});
