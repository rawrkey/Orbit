/**
 * State-derived display gates. The UI is a pure function of the backend
 * status: nothing here invents state, it only classifies what the API said.
 */

import type { WorkflowState } from '../api/types';

export function isTerminal(state: WorkflowState): boolean {
  return state === 'COMPLETED' || state === 'FAILED';
}

export function isClarifying(state: WorkflowState): boolean {
  return state === 'AWAITING_CLARIFICATION';
}

/** Actively progressing (or resumable) states — drive event polling. */
export function isActiveState(state: WorkflowState): boolean {
  return !isTerminal(state) && !isClarifying(state);
}

/**
 * POST /run is legal from every non-terminal, non-clarifying state
 * (mirrors WorkflowOrchestrator.run dispatch). Clarification states must
 * answer first — that is what makes clarification impossible to bypass.
 */
export function canRun(state: WorkflowState): boolean {
  return isActiveState(state);
}

export function phaseLabel(state: WorkflowState): string {
  switch (state) {
    case 'RECEIVED':
      return 'Received';
    case 'ANALYZING':
      return 'Analyzing goal';
    case 'AWAITING_CLARIFICATION':
      return 'Awaiting clarification';
    case 'PLANNING':
      return 'Planning';
    case 'READY':
      return 'Ready to execute';
    case 'EXECUTING':
      return 'Executing';
    case 'OBSERVING':
      return 'Observing';
    case 'AWAITING_VALIDATION':
      return 'Awaiting validation';
    case 'VALIDATING':
      return 'Validating';
    case 'REPLANNING':
      return 'Replanning';
    case 'COMPLETED':
      return 'Completed';
    case 'FAILED':
      return 'Failed';
  }
}

/** Every clarification question has a non-empty (trimmed) answer. */
export function allAnswered(
  questions: readonly string[],
  answers: Readonly<Record<string, string>>,
): boolean {
  if (questions.length === 0) return false;
  return questions.every((question) => (answers[question] ?? '').trim().length > 0);
}

/**
 * System-language for loading/transition states, translated from the REAL
 * backend state only (null for RECEIVED/terminal — nothing is inferred from
 * frontend timing, no progress percentages, no timers).
 */
export function systemPhrase(state: WorkflowState): string | null {
  switch (state) {
    case 'ANALYZING':
      return 'ANALYZING GOAL';
    case 'AWAITING_CLARIFICATION':
      return 'AWAITING INPUT';
    case 'PLANNING':
      return 'BUILDING PLAN';
    case 'READY':
      return 'PLAN READY';
    case 'EXECUTING':
      return 'EXECUTING';
    case 'OBSERVING':
      return 'OBSERVING';
    case 'AWAITING_VALIDATION':
    case 'VALIDATING':
      return 'VALIDATING';
    case 'REPLANNING':
      return 'ADAPTING';
    default:
      return null;
  }
}

/** Inline phrase for the run-in-progress line (backend state, lowercased). */
export function runPhrase(state: WorkflowState): string {
  switch (state) {
    case 'ANALYZING':
      return 'analyzing goal';
    case 'PLANNING':
      return 'building plan';
    case 'READY':
      return 'executing plan';
    case 'EXECUTING':
      return 'executing';
    case 'OBSERVING':
      return 'observing';
    case 'AWAITING_VALIDATION':
    case 'VALIDATING':
      return 'validating';
    case 'REPLANNING':
      return 'adapting';
    case 'AWAITING_CLARIFICATION':
      return 'awaiting input';
    default:
      return 'the backend is working';
  }
}
