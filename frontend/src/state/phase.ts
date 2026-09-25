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
