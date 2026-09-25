/**
 * useOrbit — the single source of truth connecting the UI to the backend.
 *
 * Invariants enforced here:
 * 1. Duplicate-run safety: a synchronous ref lock makes double-submits
 *    impossible even before React re-renders the disabled button.
 * 2. No event duplication: polls send `after=<lastSeq>`; accumulation is
 *    seq-keyed and sorted (see state/events.ts). Empty polls return the
 *    same array reference -> no re-render.
 * 3. COMPLETED is only ever displayed from a GET /workflows status — the
 *    run report alone never flips the UI into a completed view (the final
 *    status refresh is the source of truth; if it fails we stay on the
 *    last known backend state and show the error).
 * 4. Clarification cannot be bypassed: run() refuses to fire while the
 *    backend state is AWAITING_CLARIFICATION unless every question has a
 *    non-empty answer; the clarifying state can never launch a plain run.
 * 5. Browser refresh: workflow id persists in localStorage; on mount we
 *    re-fetch status + events and resume polling when the state is active.
 */

import { useCallback, useEffect, useMemo, useReducer, useRef } from 'react';

import { ApiError, NetworkError, OrbitClient, toErrorView, type ErrorView } from '../api/client';
import type {
  EventView,
  RunWorkflowResponse,
  WorkflowStatusResponse,
} from '../api/types';
import { mergeEvents } from '../state/events';
import { allAnswered, isActiveState, isClarifying } from '../state/phase';

export const STORAGE_KEY = 'orbit.workflowId';

const DEFAULT_POLL_MS = 700;

interface State {
  goal: string;
  creating: boolean;
  running: boolean;
  workflowId: string | null;
  status: WorkflowStatusResponse | null;
  events: EventView[];
  lastSeq: number;
  report: RunWorkflowResponse | null;
  error: ErrorView | null;
  notice: string | null;
  answers: Record<string, string>;
}

const initialState: State = {
  goal: '',
  creating: false,
  running: false,
  workflowId: null,
  status: null,
  events: [],
  lastSeq: -1,
  report: null,
  error: null,
  notice: null,
  answers: {},
};

type Action =
  | { type: 'GOAL'; value: string }
  | { type: 'ERROR'; error: ErrorView | null }
  | { type: 'NOTICE'; notice: string | null }
  | { type: 'CREATE_START' }
  | { type: 'CREATE_END' }
  | {
      type: 'CREATE_OK';
      workflowId: string;
      status: WorkflowStatusResponse;
      events: EventView[];
    }
  | { type: 'RUN_START' }
  | { type: 'RUN_SUCCESS'; report: RunWorkflowResponse }
  | { type: 'RUN_END' }
  | { type: 'STATUS'; status: WorkflowStatusResponse }
  | { type: 'EVENTS'; events: EventView[] }
  | { type: 'ANSWER'; question: string; value: string }
  | { type: 'CLEAR_WORKFLOW' }
  | { type: 'RESET' };

function accumulate(state: State, incoming: readonly EventView[]): Pick<State, 'events' | 'lastSeq'> {
  const merged = mergeEvents(state.events, incoming, state.lastSeq);
  return { events: merged.events, lastSeq: merged.lastSeq };
}

function reducer(state: State, action: Action): State {
  switch (action.type) {
    case 'GOAL':
      return { ...state, goal: action.value };
    case 'ERROR':
      return { ...state, error: action.error };
    case 'NOTICE':
      return { ...state, notice: action.notice };
    case 'CREATE_START':
      return { ...state, creating: true, error: null, notice: null };
    case 'CREATE_END':
      return { ...state, creating: false };
    case 'CREATE_OK': {
      const base: State = {
        ...state,
        creating: false,
        workflowId: action.workflowId,
        status: action.status,
        report: null,
        error: null,
        answers: {},
        events: [],
        lastSeq: -1,
      };
      return { ...base, ...accumulate(base, action.events) };
    }
    case 'RUN_START':
      return { ...state, running: true, error: null, notice: null, report: null };
    case 'RUN_SUCCESS':
      return { ...state, report: action.report };
    case 'RUN_END':
      return { ...state, running: false };
    case 'STATUS':
      return { ...state, status: action.status };
    case 'EVENTS':
      return { ...state, ...accumulate(state, action.events) };
    case 'ANSWER':
      return { ...state, answers: { ...state.answers, [action.question]: action.value } };
    case 'CLEAR_WORKFLOW':
      return {
        ...state,
        workflowId: null,
        status: null,
        events: [],
        lastSeq: -1,
        report: null,
        answers: {},
      };
    case 'RESET':
      return { ...initialState };
  }
}

export interface UseOrbitOptions {
  pollMs?: number;
  client?: OrbitClient;
}

export interface OrbitController {
  goal: string;
  creating: boolean;
  running: boolean;
  workflowId: string | null;
  status: WorkflowStatusResponse | null;
  events: EventView[];
  report: RunWorkflowResponse | null;
  error: ErrorView | null;
  notice: string | null;
  answers: Record<string, string>;
  setGoal: (value: string) => void;
  setAnswer: (question: string, value: string) => void;
  create: () => void;
  run: () => void;
  dismissError: () => void;
  reset: () => void;
}

export function useOrbit(options: UseOrbitOptions = {}): OrbitController {
  const pollMs = options.pollMs ?? DEFAULT_POLL_MS;
  const clientRef = useRef<OrbitClient | null>(null);
  clientRef.current ??= options.client ?? new OrbitClient();
  const client = clientRef.current;

  const [state, dispatch] = useReducer(reducer, initialState);

  // Mirrors for lock-guarded async callbacks (always read freshest render).
  const stateRef = useRef(state);
  stateRef.current = state;
  const runLock = useRef(false);
  const createLock = useRef(false);
  const pollBusy = useRef(false);

  // ------------------------------------------------------------------
  // Refresh after create / restore / run: status + events, always.
  // ------------------------------------------------------------------
  const refresh = useCallback(
    async (workflowId: string): Promise<void> => {
      const [status, page] = await Promise.all([
        client.getWorkflow(workflowId),
        client.getEvents(workflowId),
      ]);
      dispatch({ type: 'STATUS', status });
      dispatch({ type: 'EVENTS', events: page.events });
    },
    [client],
  );

  // ------------------------------------------------------------------
  // Restore after browser refresh (localStorage holds the workflow id).
  // ------------------------------------------------------------------
  useEffect(() => {
    const saved = window.localStorage.getItem(STORAGE_KEY);
    if (!saved) return;
    let cancelled = false;
    void (async () => {
      try {
        const [status, page] = await Promise.all([
          client.getWorkflow(saved),
          client.getEvents(saved),
        ]);
        if (cancelled) return;
        dispatch({ type: 'CREATE_OK', workflowId: saved, status, events: page.events });
        dispatch({ type: 'NOTICE', notice: 'Restored workflow from a previous session.' });
      } catch (error) {
        if (cancelled) return;
        if (error instanceof ApiError && error.status === 404) {
          window.localStorage.removeItem(STORAGE_KEY);
          dispatch({ type: 'CLEAR_WORKFLOW' });
          dispatch({ type: 'NOTICE', notice: 'Saved workflow no longer exists.' });
        } else {
          dispatch({ type: 'ERROR', error: toErrorView(error) });
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [client]);

  // ------------------------------------------------------------------
  // Polling: only while a run is in flight or the backend state is active.
  // (RECEIVED with no progress is idle — nothing changes server-side.)
  // ------------------------------------------------------------------
  const shouldPoll =
    state.running ||
    (state.status !== null &&
      isActiveState(state.status.state) &&
      (state.status.state !== 'RECEIVED' || state.events.length > 1));

  const pollOnce = useCallback(async (): Promise<void> => {
    const current = stateRef.current;
    const workflowId = current.workflowId;
    if (!workflowId || pollBusy.current) return;
    pollBusy.current = true;
    try {
      const [status, page] = await Promise.all([
        client.getWorkflow(workflowId),
        client.getEvents(workflowId, current.lastSeq >= 0 ? current.lastSeq : undefined),
      ]);
      dispatch({ type: 'STATUS', status });
      dispatch({ type: 'EVENTS', events: page.events });
    } catch (error) {
      if (error instanceof ApiError && error.status === 404) {
        window.localStorage.removeItem(STORAGE_KEY);
        dispatch({ type: 'ERROR', error: toErrorView(error) });
        dispatch({ type: 'CLEAR_WORKFLOW' });
      } else if (error instanceof NetworkError) {
        dispatch({ type: 'NOTICE', notice: 'Connection issue while polling — retrying…' });
      } else {
        dispatch({ type: 'ERROR', error: toErrorView(error) });
      }
    } finally {
      pollBusy.current = false;
    }
  }, [client]);

  useEffect(() => {
    if (!shouldPoll) return;
    const timer = window.setInterval(() => void pollOnce(), pollMs);
    return () => window.clearInterval(timer);
  }, [shouldPoll, pollMs, pollOnce]);

  // ------------------------------------------------------------------
  // Actions
  // ------------------------------------------------------------------
  const setGoal = useCallback((value: string) => dispatch({ type: 'GOAL', value }), []);
  const setAnswer = useCallback(
    (question: string, value: string) => dispatch({ type: 'ANSWER', question, value }),
    [],
  );
  const dismissError = useCallback(() => dispatch({ type: 'ERROR', error: null }), []);

  const create = useCallback((): void => {
    const current = stateRef.current;
    const goal = current.goal.trim();
    if (createLock.current) return;
    if (goal.length === 0) {
      dispatch({
        type: 'ERROR',
        error: { kind: 'client', message: 'Goal must not be blank.' },
      });
      return;
    }
    createLock.current = true;
    dispatch({ type: 'CREATE_START' });
    void (async () => {
      try {
        const created = await client.createWorkflow(goal);
        window.localStorage.setItem(STORAGE_KEY, created.workflow_id);
        const [status, page] = await Promise.all([
          client.getWorkflow(created.workflow_id),
          client.getEvents(created.workflow_id),
        ]);
        dispatch({
          type: 'CREATE_OK',
          workflowId: created.workflow_id,
          status,
          events: page.events,
        });
      } catch (error) {
        dispatch({ type: 'ERROR', error: toErrorView(error) });
      } finally {
        createLock.current = false;
        dispatch({ type: 'CREATE_END' });
      }
    })();
  }, [client]);

  const run = useCallback((): void => {
    const current = stateRef.current;
    const workflowId = current.workflowId;
    if (runLock.current || !workflowId) return;

    // Clarification bypass guard: a plain run must never fire while the
    // backend awaits answers; submitting answers is the only way out.
    const clarifying = current.status !== null && isClarifying(current.status.state);
    if (clarifying) {
      const questions = current.status?.clarification_questions ?? [];
      if (!allAnswered(questions, current.answers)) return;
    }

    runLock.current = true;
    dispatch({ type: 'RUN_START' });
    void (async () => {
      try {
        const report = await client.runWorkflow(
          workflowId,
          clarifying ? current.answers : undefined,
        );
        dispatch({ type: 'RUN_SUCCESS', report });
        try {
          await refresh(workflowId);
        } catch (error) {
          // The run succeeded but the source-of-truth refresh failed:
          // surface the error and keep the LAST KNOWN backend status —
          // the UI must not claim COMPLETED without a backend status.
          dispatch({ type: 'ERROR', error: toErrorView(error) });
        }
      } catch (error) {
        dispatch({ type: 'ERROR', error: toErrorView(error) });
        if (error instanceof ApiError && error.status === 409) {
          // Rejected before mutation (missing answers): re-sync status so
          // the clarification form reflects backend truth.
          try {
            const status = await client.getWorkflow(workflowId);
            dispatch({ type: 'STATUS', status });
          } catch {
            // keep the original 409 error displayed
          }
        }
      } finally {
        runLock.current = false;
        dispatch({ type: 'RUN_END' });
      }
    })();
  }, [client, refresh]);

  const reset = useCallback((): void => {
    window.localStorage.removeItem(STORAGE_KEY);
    dispatch({ type: 'RESET' });
  }, []);

  return useMemo(
    () => ({
      goal: state.goal,
      creating: state.creating,
      running: state.running,
      workflowId: state.workflowId,
      status: state.status,
      events: state.events,
      report: state.report,
      error: state.error,
      notice: state.notice,
      answers: state.answers,
      setGoal,
      setAnswer,
      create,
      run,
      dismissError,
      reset,
    }),
    [state, setGoal, setAnswer, create, run, dismissError, reset],
  );
}
