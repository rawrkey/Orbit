/**
 * TypeScript contracts for the ORBIT backend API.
 *
 * Every shape mirrors app/api/schemas.py exactly; fixtures under
 * ./fixtures are generated from the live backend (scripts/generate_fixtures.py)
 * and checked against these types at compile time + key-set runtime tests.
 */

export const WORKFLOW_STATES = [
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
] as const;
export type WorkflowState = (typeof WORKFLOW_STATES)[number];

export const EVENT_TYPES = [
  'workflow.started',
  'goal.analyzed',
  'clarification.requested',
  'clarification.answered',
  'workflow.planned',
  'plan.validation_failed',
  'state.changed',
  'execution.started',
  'execution.finished',
  'execution.failed',
  'task.started',
  'tool.decision_proposed',
  'tool.called',
  'tool.completed',
  'tool.failed',
  'observation.created',
  'validation.started',
  'validation.passed',
  'validation.failed',
  'replanning.started',
  'replan.proposed',
  'replan.accepted',
  'replan.rejected',
  'replanning.exhausted',
  'workflow.replanned',
  'task.completed',
  'synthesis.completed',
  'workflow.completed',
] as const;
export type EventType = (typeof EVENT_TYPES)[number];

export const TERMINATIONS = [
  'validated',
  'already_terminal',
  'analysis_error',
  'awaiting_clarification',
  'execution_failed',
  'infeasible',
  'planning_error',
  'replan_error',
  'replan_exhausted',
  'replan_invalid',
  'synthesis_error',
  'validation_error',
] as const;
export type Termination = (typeof TERMINATIONS)[number];

export type JsonValue =
  | string
  | number
  | boolean
  | null
  | JsonValue[]
  | { [key: string]: JsonValue };

/** FailureContext (app/state/models.py) */
export interface FailureContext {
  stage: WorkflowState;
  error_code: string;
  detail: string;
  task_id: string | null;
}

/** POST /workflows request */
export interface CreateWorkflowRequest {
  goal: string;
}

/** POST /workflows response (201) */
export interface CreateWorkflowResponse {
  workflow_id: string;
  state: WorkflowState;
}

/** GET /workflows/{id} response */
export interface WorkflowStatusResponse {
  workflow_id: string;
  goal: string;
  state: WorkflowState;
  replan_count: number;
  created_at: string;
  updated_at: string;
  task_count: number;
  observation_count: number;
  event_count: number;
  plan_version: number | null;
  clarification_questions: string[];
  final_answer: string | null;
  failure: FailureContext | null;
}

/** POST /workflows/{id}/run request */
export interface RunWorkflowRequest {
  answers?: Record<string, string>;
}

/** POST /workflows/{id}/run response */
export interface RunWorkflowResponse {
  workflow_id: string;
  success: boolean;
  final_state: WorkflowState;
  executions: number;
  validations: number;
  replans: number;
  termination: Termination;
  failure: FailureContext | null;
}

/** One event as served by GET /workflows/{id}/events */
export interface EventView {
  seq: number;
  timestamp: string;
  type: EventType;
  state: WorkflowState;
  task_id: string | null;
  decision_summary: string | null;
  payload: Record<string, JsonValue>;
}

/** GET /workflows/{id}/events response */
export interface EventsResponse {
  workflow_id: string;
  events: EventView[];
  count: number;
}
