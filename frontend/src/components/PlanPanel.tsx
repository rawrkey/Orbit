import { useMemo } from 'react';
import type { CSSProperties } from 'react';

import type { EventView } from '../api/types';
import { derivePlan, deriveReplanChain, type ChainStep, type PlanTask } from '../state/eventVisual';

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

function AdaptChain({ steps }: { steps: ChainStep[] }) {
  return (
    <div className="adapt">
      <div className="panel-head">
        <h3 className="band-label band-label--accent">Adapt</h3>
        <span className="panel-meta mono">replan</span>
      </div>
      <ol className="chain">
        {steps.map((step, index) => (
          <li
            key={step.label}
            data-state={step.state}
            className={`chain-step chain-step--${step.state}`}
            style={{ '--i': index } as CSSProperties}
          >
            <span className="chain-label">{step.label}</span>
            {step.detail && <span className="chain-detail mono">{step.detail}</span>}
          </li>
        ))}
      </ol>
    </div>
  );
}

/**
 * The PLAN column: current plan recovered from the event log, task progress
 * glyphs (✓ ◉ ○ !), and — when the backend adapted — the replan chain.
 */
export function PlanPanel({ events, active }: PlanPanelProps) {
  const plan = useMemo(() => derivePlan(events), [events]);
  const chain = useMemo(() => deriveReplanChain(events), [events]);
  const done = plan.tasks.filter((task) => task.status === 'done').length;
  const total = plan.tasks.length;

  return (
    <section aria-label="plan" className="band plan-panel">
      <div className="panel-head">
        <h2 className="band-label">Plan</h2>
        <span className="panel-meta mono">
          {plan.version !== null ? `v${plan.version}` : '—'}
          {total > 0 ? ` · ${done}/${total}` : ''}
        </span>
      </div>

      {total > 0 ? (
        <ol className="tasks">
          {plan.tasks.map((task) => (
            <TaskRow key={task.id} task={task} active={active} />
          ))}
        </ol>
      ) : (
        <p className="empty mono">No plan recorded yet.</p>
      )}

      {chain && <AdaptChain steps={chain} />}
    </section>
  );
}
