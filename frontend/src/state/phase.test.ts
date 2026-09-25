/**
 * Every backend state must be handled: labels exist for all 12 states,
 * run gating mirrors the orchestrator dispatch (never terminal, never
 * while awaiting clarification), and the clarifying state always demands
 * answers before anything can run.
 */

import { describe, expect, it } from 'vitest';

import { WORKFLOW_STATES, type WorkflowState } from '../api/types';
import { allAnswered, canRun, isClarifying, isActiveState, isTerminal, phaseLabel } from './phase';

describe('phase classification', () => {
  it('covers every WorkflowState member with a non-empty label', () => {
    for (const state of WORKFLOW_STATES) {
      expect(phaseLabel(state)).toBeTruthy();
    }
    expect(WORKFLOW_STATES).toHaveLength(12);
  });

  it('marks exactly COMPLETED and FAILED as terminal', () => {
    const terminals = WORKFLOW_STATES.filter(isTerminal);
    expect(terminals).toEqual(['COMPLETED', 'FAILED']);
  });

  it('marks exactly AWAITING_CLARIFICATION as clarifying', () => {
    const clarifying = WORKFLOW_STATES.filter(isClarifying);
    expect(clarifying).toEqual(['AWAITING_CLARIFICATION']);
  });

  it('allows run from every state except terminal and clarifying', () => {
    const runnable = WORKFLOW_STATES.filter(canRun);
    expect(runnable).toEqual([
      'RECEIVED',
      'ANALYZING',
      'PLANNING',
      'READY',
      'EXECUTING',
      'OBSERVING',
      'VALIDATING',
      'AWAITING_VALIDATION',
      'REPLANNING',
    ]);
  });

  it('COMPLETED and FAILED can never run or poll as active', () => {
    for (const state of ['COMPLETED', 'FAILED'] as const) {
      expect(canRun(state)).toBe(false);
      expect(isActiveState(state)).toBe(false);
    }
    expect(isActiveState('AWAITING_CLARIFICATION')).toBe(false);
    expect(isActiveState('EXECUTING')).toBe(true);
  });
});

describe('allAnswered', () => {
  const questions = ['Please provide: dates', 'Please provide: budget'];

  it('is false while any answer is missing or blank', () => {
    expect(allAnswered(questions, {})).toBe(false);
    expect(allAnswered(questions, { 'Please provide: dates': 'Dec 1-4' })).toBe(false);
    expect(
      allAnswered(questions, {
        'Please provide: dates': 'Dec 1-4',
        'Please provide: budget': '   ',
      }),
    ).toBe(false);
  });

  it('is true only when every question has a non-empty answer', () => {
    expect(
      allAnswered(questions, {
        'Please provide: dates': 'Dec 1-4',
        'Please provide: budget': '8000',
      }),
    ).toBe(true);
  });

  it('is false for an empty question list (nothing to satisfy)', () => {
    expect(allAnswered([], {})).toBe(false);
  });
});

describe('state coverage', () => {
  it('the state list never drifts from known states', () => {
    const expected: WorkflowState[] = [
      'RECEIVED',
      'ANALYZING',
      'AWAITING_CLARIFICATION',
      'PLANNING',
      'READY',
      'EXECUTING',
      'OBSERVING',
      'VALIDATING',
      'AWAITING_VALIDATION',
      'REPLANNING',
      'COMPLETED',
      'FAILED',
    ];
    expect([...WORKFLOW_STATES]).toEqual(expected);
  });
});
