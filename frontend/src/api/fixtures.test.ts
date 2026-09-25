/**
 * Fixture contract tests: the TypeScript types must match the ACTUAL
 * backend responses (fixtures are generated from the live API by
 * scripts/generate_fixtures.py — never hand-written).
 *
 * JSON imports widen string literals, so compile-time `satisfies` cannot
 * verify them. Instead every fixture goes through the runtime parsers in
 * contract.ts (exact key sets, primitive types, enum membership) and is
 * then asserted against the exact values the flow depends on.
 */

import { describe, expect, it } from 'vitest';

import { parseCreate, parseEvents, parseRun, parseStatus } from './contract';
import { EVENT_TYPES, WORKFLOW_STATES } from './types';

import create201 from './fixtures/create_201.json';
import error404 from './fixtures/error_404.json';
import error409 from './fixtures/error_409.json';
import error422 from './fixtures/error_422.json';
import eventsDemo from './fixtures/events_demo.json';
import eventsReceived from './fixtures/events_received.json';
import runAlreadyTerminal from './fixtures/run_already_terminal.json';
import runAwaiting from './fixtures/run_awaiting.json';
import runFailed from './fixtures/run_failed.json';
import runValidated from './fixtures/run_validated.json';
import statusAwaiting from './fixtures/status_awaiting_clarification.json';
import statusCompleted from './fixtures/status_completed.json';
import statusExecuting from './fixtures/status_executing.json';
import statusFailed from './fixtures/status_failed.json';
import statusReceived from './fixtures/status_received.json';
import statusReplanning from './fixtures/status_replanning.json';
import statusValidating from './fixtures/status_validating.json';

describe('status fixtures', () => {
  const statuses = {
    received: parseStatus(statusReceived),
    completed: parseStatus(statusCompleted),
    awaiting: parseStatus(statusAwaiting),
    failed: parseStatus(statusFailed),
    executing: parseStatus(statusExecuting),
    validating: parseStatus(statusValidating),
    replanning: parseStatus(statusReplanning),
  };

  it('produces the exact states the flow needs', () => {
    expect(statuses.received.state).toBe('RECEIVED');
    expect(statuses.completed.state).toBe('COMPLETED');
    expect(statuses.awaiting.state).toBe('AWAITING_CLARIFICATION');
    expect(statuses.failed.state).toBe('FAILED');
    expect(statuses.executing.state).toBe('EXECUTING');
    expect(statuses.validating.state).toBe('VALIDATING');
    expect(statuses.replanning.state).toBe('REPLANNING');
    expect(WORKFLOW_STATES).toContain(statuses.replanning.state);
  });

  it('COMPLETED carries a final answer, plan version, and no failure', () => {
    expect(statuses.completed.final_answer).toContain('7,500');
    expect(statuses.completed.failure).toBeNull();
    expect(statuses.completed.plan_version).toBe(2);
    expect(statuses.completed.event_count).toBe(eventsDemo.events.length);
  });

  it('AWAITING_CLARIFICATION exposes its question and no answer', () => {
    expect(statuses.awaiting.clarification_questions).toEqual([
      'Please provide: travel dates',
    ]);
    expect(statuses.awaiting.final_answer).toBeNull();
  });

  it('FAILED carries a structured failure context', () => {
    expect(statuses.failed.failure).not.toBeNull();
    expect(statuses.failed.failure?.stage).toBe('ANALYZING');
    expect(statuses.failed.failure?.error_code).toBe('LLM_ERROR');
    expect(statuses.failed.final_answer).toBeNull();
  });

  it('mid-run states have no final answer (results cannot leak early)', () => {
    for (const status of [statuses.executing, statuses.validating, statuses.replanning]) {
      expect(status.final_answer).toBeNull();
      expect(status.failure).toBeNull();
    }
  });

  it('REPLANNING reflects consumed replan budget and stored events', () => {
    expect(statuses.replanning.replan_count).toBeGreaterThanOrEqual(1);
    expect(statuses.replanning.event_count).toBeGreaterThan(0);
  });
});

describe('run fixtures', () => {
  const reports = {
    validated: parseRun(runValidated),
    awaiting: parseRun(runAwaiting),
    failed: parseRun(runFailed),
    alreadyTerminal: parseRun(runAlreadyTerminal),
  };

  it('validated: success, COMPLETED, multiple passes, one replan', () => {
    expect(reports.validated.success).toBe(true);
    expect(reports.validated.termination).toBe('validated');
    expect(reports.validated.final_state).toBe('COMPLETED');
    expect(reports.validated.executions).toBeGreaterThanOrEqual(2);
    expect(reports.validated.validations).toBeGreaterThanOrEqual(2);
    expect(reports.validated.replans).toBeGreaterThanOrEqual(1);
    expect(reports.validated.failure).toBeNull();
  });

  it('awaiting: not successful, never claims COMPLETED', () => {
    expect(reports.awaiting.success).toBe(false);
    expect(reports.awaiting.termination).toBe('awaiting_clarification');
    expect(reports.awaiting.final_state).toBe('AWAITING_CLARIFICATION');
  });

  it('failed: failure context present', () => {
    expect(reports.failed.success).toBe(false);
    expect(reports.failed.failure).not.toBeNull();
    expect(reports.failed.failure?.error_code).toBeTruthy();
  });

  it('already_terminal rerun reports success with zero new passes', () => {
    expect(reports.alreadyTerminal.termination).toBe('already_terminal');
    expect(reports.alreadyTerminal.success).toBe(true);
    expect(reports.alreadyTerminal.executions).toBe(0);
    expect(reports.alreadyTerminal.validations).toBe(0);
  });
});

describe('event fixtures', () => {
  const demo = parseEvents(eventsDemo);
  const received = parseEvents(eventsReceived);

  it('pages agree with their event arrays and use valid enums', () => {
    expect(demo.count).toBe(demo.events.length);
    expect(received.count).toBe(received.events.length);
    for (const event of [...demo.events, ...received.events]) {
      expect(EVENT_TYPES).toContain(event.type);
      expect(WORKFLOW_STATES).toContain(event.state);
      expect(typeof event.seq).toBe('number');
    }
  });

  it('event seqs are contiguous from 0 and strictly increasing', () => {
    const seqs = demo.events.map((event) => event.seq);
    expect(seqs).toEqual([...seqs].sort((a, b) => a - b));
    expect(new Set(seqs).size).toBe(seqs.length);
    expect(seqs[0]).toBe(0);
    expect(seqs[seqs.length - 1]).toBe(seqs.length - 1);
  });

  it('the demo story shows FAIL -> replan -> PASS -> synthesis -> completed', () => {
    const types = demo.events.map((event) => event.type);
    expect(types).toContain('validation.failed');
    expect(types).toContain('replan.accepted');
    expect(types).toContain('validation.passed');
    expect(types).toContain('synthesis.completed');
    expect(types[types.length - 1]).toBe('workflow.completed');
    expect(types.indexOf('validation.failed')).toBeLessThan(types.indexOf('replan.accepted'));
    expect(types.indexOf('replan.accepted')).toBeLessThan(types.indexOf('validation.passed'));
    expect(types.indexOf('validation.passed')).toBeLessThan(
      types.indexOf('synthesis.completed'),
    );
  });
});

describe('create and error fixtures', () => {
  it('create response parses with the exact backend shape', () => {
    const created = parseCreate(create201);
    expect(created.state).toBe('RECEIVED');
    expect(created.workflow_id).toMatch(/^[0-9a-f]{32}$/);
  });

  it('404/409 carry a string detail', () => {
    expect(Object.keys(error404)).toEqual(['detail']);
    expect(typeof error404.detail).toBe('string');
    expect(Object.keys(error409)).toEqual(['detail']);
    expect(error409.detail).toContain('missing answers');
  });

  it('422 carries the FastAPI issue array', () => {
    expect(Object.keys(error422)).toEqual(['detail']);
    expect(Array.isArray(error422.detail)).toBe(true);
    expect(error422.detail[0]?.loc).toEqual(['body', 'goal']);
    expect(error422.detail[0]?.msg).toContain('at least 1 character');
  });
});
