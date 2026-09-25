import type { RunWorkflowResponse, WorkflowStatusResponse } from '../api/types';
import { canRun, isTerminal, phaseLabel } from '../state/phase';

interface StatusPanelProps {
  status: WorkflowStatusResponse;
  report: RunWorkflowResponse | null;
  running: boolean;
  onRun: () => void;
  onReset: () => void;
}

/**
 * The persistent operations console: workflow state + compact metadata +
 * run controls. Failure context lives in its own VALIDATION band (App).
 */
export function StatusPanel({ status, report, running, onRun, onReset }: StatusPanelProps) {
  const showRunButton = canRun(status.state);

  return (
    <section aria-label="status-panel" data-testid="status-panel" className="status-console">
      <div className="panel-head">
        <h2 className="band-label">Workflow status</h2>
        <span className="panel-meta mono">wf {status.workflow_id.slice(0, 8)}</span>
      </div>

      <div className="state-value" data-testid="state-value" data-state={status.state}>
        <span className="state-dot" data-state={status.state} aria-hidden="true" />
        <span className="state-label">{phaseLabel(status.state)}</span>{' '}
        <code className="state-code">({status.state})</code>
      </div>

      <dl className="meta-grid mono">
        <dt>Tasks</dt>
        <dd>{status.task_count}</dd>
        <dt>Observations</dt>
        <dd>{status.observation_count}</dd>
        <dt>Events</dt>
        <dd>{status.event_count}</dd>
        <dt>Plan</dt>
        <dd>{status.plan_version !== null ? `v${status.plan_version}` : '—'}</dd>
        <dt>Replans</dt>
        <dd>{status.replan_count}</dd>
      </dl>

      {running && (
        <p role="status" data-testid="run-in-progress" className="run-live">
          <span className="live-dot" aria-hidden="true" />
          Run in progress — the backend is working…
        </p>
      )}

      {report && !running && (
        <p data-testid="run-report" className="run-report mono">
          Run finished: <strong>{report.termination}</strong> · {report.executions} execution
          pass{report.executions === 1 ? '' : 'es'} · {report.validations} validation
          {report.validations === 1 ? '' : 's'} · {report.replans} replan
          {report.replans === 1 ? '' : 's'}
        </p>
      )}

      <div className="console-actions">
        {showRunButton && (
          <button
            type="button"
            onClick={onRun}
            disabled={running}
            data-testid="run-button"
            className="btn btn--primary"
          >
            {running
              ? 'Running…'
              : status.state === 'RECEIVED'
                ? 'Run workflow'
                : 'Resume run'}
          </button>
        )}
        {!running && (
          <button
            type="button"
            onClick={onReset}
            data-testid="new-goal-button"
            className="btn btn--ghost"
          >
            {isTerminal(status.state) ? 'New goal' : 'Discard and start over'}
          </button>
        )}
      </div>
    </section>
  );
}
