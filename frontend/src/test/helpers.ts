import type {
  CreateWorkflowResponse,
  EventsResponse,
  EventView,
  RunWorkflowResponse,
  WorkflowState,
  WorkflowStatusResponse,
} from '../api/types';

const BASE_STATUS: WorkflowStatusResponse = {
  workflow_id: 'wf-test',
  goal: 'Test goal',
  state: 'RECEIVED',
  replan_count: 0,
  created_at: '2026-01-01T00:00:00Z',
  updated_at: '2026-01-01T00:00:00Z',
  task_count: 0,
  observation_count: 0,
  event_count: 1,
  plan_version: null,
  clarification_questions: [],
  final_answer: null,
  failure: null,
};

/** Full, type-correct status object for any backend state. */
export function makeStatus(
  state: WorkflowState,
  overrides: Partial<WorkflowStatusResponse> = {},
): WorkflowStatusResponse {
  return { ...BASE_STATUS, state, ...overrides };
}

export const CREATED: CreateWorkflowResponse = {
  workflow_id: 'wf-test',
  state: 'RECEIVED',
};

export function makeEvent(seq: number, overrides: Partial<EventView> = {}): EventView {
  return {
    seq,
    timestamp: '2026-01-01T00:00:00Z',
    type: 'state.changed',
    state: 'RECEIVED',
    task_id: null,
    decision_summary: `event ${seq}`,
    payload: {},
    ...overrides,
  };
}

export function makePage(
  events: EventView[],
  workflowId = 'wf-test',
): EventsResponse {
  return { workflow_id: workflowId, events, count: events.length };
}

export const VALIDATED_RUN: RunWorkflowResponse = {
  workflow_id: 'wf-test',
  success: true,
  final_state: 'COMPLETED',
  executions: 2,
  validations: 2,
  replans: 1,
  termination: 'validated',
  failure: null,
};
