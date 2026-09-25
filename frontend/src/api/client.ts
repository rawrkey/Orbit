/**
 * HTTP client for the four ORBIT endpoints.
 *
 * Error contract:
 * - NetworkError  -> fetch itself failed (server unreachable / offline)
 * - ApiError      -> HTTP error; 404 unknown workflow, 409 controller
 *                    rejection (e.g. missing clarification answers),
 *                    422 FastAPI request validation (issues[] parsed)
 *
 * The client never retries on its own — duplicate-submission safety is the
 * caller's job (see useOrbit's synchronous run lock).
 */

import type {
  CreateWorkflowRequest,
  CreateWorkflowResponse,
  EventsResponse,
  RunWorkflowRequest,
  RunWorkflowResponse,
  WorkflowStatusResponse,
} from './types';

export class ApiError extends Error {
  readonly status: number;
  readonly issues: readonly string[];

  constructor(status: number, message: string, issues: readonly string[] = []) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.issues = issues;
  }
}

export class NetworkError extends Error {
  constructor(message = 'Could not reach the ORBIT API. Is the backend running?') {
    super(message);
    this.name = 'NetworkError';
  }
}

export type ErrorView =
  | { kind: 'network'; message: string }
  | { kind: 'api'; status: number; message: string; issues: readonly string[] }
  | { kind: 'client'; message: string };

export function toErrorView(error: unknown): ErrorView {
  if (error instanceof ApiError) {
    return { kind: 'api', status: error.status, message: error.message, issues: error.issues };
  }
  if (error instanceof NetworkError) {
    return { kind: 'network', message: error.message };
  }
  return { kind: 'client', message: error instanceof Error ? error.message : 'Unexpected error.' };
}

interface ParsedDetail {
  message: string;
  issues: string[];
}

function parseDetail(detail: unknown): ParsedDetail {
  if (typeof detail === 'string') {
    return { message: detail, issues: [] };
  }
  if (Array.isArray(detail)) {
    const issues = detail.map((item): string => {
      if (item !== null && typeof item === 'object' && 'msg' in item) {
        const msg = (item as { msg: unknown }).msg;
        if (typeof msg === 'string') return msg;
      }
      return JSON.stringify(item);
    });
    return { message: issues.join(' | ') || 'Request rejected.', issues };
  }
  if (detail !== null && typeof detail === 'object') {
    return { message: JSON.stringify(detail), issues: [] };
  }
  return { message: 'Request failed.', issues: [] };
}

export class OrbitClient {
  private readonly baseUrl: string;

  constructor(baseUrl = '') {
    this.baseUrl = baseUrl;
  }

  private async request<T>(path: string, init?: RequestInit): Promise<T> {
    let response: Response;
    try {
      response = await fetch(this.baseUrl + path, {
        ...init,
        headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
      });
    } catch {
      throw new NetworkError();
    }

    if (!response.ok) {
      let detail: unknown;
      try {
        const body = (await response.json()) as { detail?: unknown };
        detail = body.detail;
      } catch {
        detail = undefined;
      }
      const { message, issues } = parseDetail(detail);
      throw new ApiError(
        response.status,
        message || `Request failed with status ${response.status}.`,
        issues,
      );
    }

    return (await response.json()) as T;
  }

  async createWorkflow(goal: string): Promise<CreateWorkflowResponse> {
    const body: CreateWorkflowRequest = { goal };
    return this.request<CreateWorkflowResponse>('/workflows', {
      method: 'POST',
      body: JSON.stringify(body),
    });
  }

  async getWorkflow(workflowId: string): Promise<WorkflowStatusResponse> {
    return this.request<WorkflowStatusResponse>(
      `/workflows/${encodeURIComponent(workflowId)}`,
    );
  }

  async runWorkflow(
    workflowId: string,
    answers?: Record<string, string>,
  ): Promise<RunWorkflowResponse> {
    const body: RunWorkflowRequest = { answers: answers ?? {} };
    return this.request<RunWorkflowResponse>(
      `/workflows/${encodeURIComponent(workflowId)}/run`,
      { method: 'POST', body: JSON.stringify(body) },
    );
  }

  async getEvents(workflowId: string, after?: number): Promise<EventsResponse> {
    const query = after === undefined ? '' : `?after=${after}`;
    return this.request<EventsResponse>(
      `/workflows/${encodeURIComponent(workflowId)}/events${query}`,
    );
  }
}
