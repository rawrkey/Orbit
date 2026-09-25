import { Fragment, useMemo } from 'react';

import type { EventView } from '../api/types';
import { derivePlan, derivePlanHistory, type PlanTask } from '../state/eventVisual';

interface PlanPanelProps {
  events: readonly EventView[];
  /** Workflow is live — gates the subtle active-task pulse. */
  active: boolean;
}

const GLYPHS: Record<PlanTask['status'], string> = {
  done: '✓',
  active: '◉',
  pending: '○',
  failed: '!',
};

const STATUS_WORDS: Record<PlanTask['status'], string> = {
  done: 'completed',
  active: 'in progress',
  pending: 'pending',
  failed: 'failed',
};

function TaskRow({ task, active }: { task: PlanTask; active: boolean }) {
  return (
    <li
      data-task-id={task.id}
      data-status={task.status}
      className={`task task--${task.status}${task.status === 'active' && active ? ' is-live' : ''}`}
    >
      <span className="task-glyph" aria-hidden="true">
        {GLYPHS[task.status]}
      </span>
      <span className="sr-only">{STATUS_WORDS[task.status]}: </span>
      <span className="task-title">{task.title ?? task.id}</span>
      <code className="task-id">{task.id}</code>
    </li>
  );
}

/**
 * The PLAN column: real plan recovered from the event log, task progress
 * glyphs (✓ ◉ ○ !), and — once a replan happened — the revision history
 * (superseded plan → what the adaptation changed → current plan below).
 */
export function PlanPanel({ events, active }: PlanPanelProps) {
  const plan = useMemo(() => derivePlan(events), [events]);
  const history = useMemo(() => derivePlanHistory(events), [events]);
  const done = plan.tasks.filter((task) => task.status === 'done').length;
  const total = plan.tasks.length;
  const superseded = history ? history.revisions.slice(0, -1) : [];

  return (
    <section aria-label="plan" className="band plan-panel">
      <div className="panel-head">
        <h2 className="band-label">Plan</h2>
        <span className="panel-meta mono">
          {plan.version !== null ? `v${plan.version}` : '—'}
          {total > 0 ? ` · ${done}/${total}` : ''}
        </span>
      </div>

      {superseded.length > 0 && history && (
        <div data-testid="plan-history" className="plan-history">
          {superseded.map((revision, index) => {
            const link = history.links[index];
            return (
              <Fragment key={revision.version}>
                <div className="plan-card">
                  <div className="plan-card-head">
                    <span className="plan-card-name">Plan v{revision.version}</span>
                    <span className="plan-card-tag mono">superseded</span>
                  </div>
                  <p className="plan-card-meta mono">
                    {revision.completed}/{revision.total} tasks completed
                    {revision.constraintFailures > 0 &&
                      ` · ${revision.constraintFailures} constraint${
                        revision.constraintFailures === 1 ? '' : 's'
                      } failed`}
                  </p>
                </div>
                {link && (
                  <p className="plan-adapt mono" data-testid="plan-adapt-link">
                    <span aria-hidden="true">↓</span> Adapted · v{link.from} → v{link.to} —{' '}
                    {link.preserved} preserved · {link.revised} revised · {link.added} new
                  </p>
                )}
              </Fragment>
            );
          })}
        </div>
      )}

      {total > 0 ? (
        <ol className="tasks">
          {plan.tasks.map((task) => (
            <TaskRow key={task.id} task={task} active={active} />
          ))}
        </ol>
      ) : (
        <p className="empty mono">No plan recorded yet.</p>
      )}
    </section>
  );
}
