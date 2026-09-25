import { useMemo } from 'react';

import type { EventView, WorkflowStatusResponse } from '../api/types';
import { deriveSynthesis, deriveValidation } from '../state/eventVisual';

interface ResultPanelProps {
  status: WorkflowStatusResponse;
  events: readonly EventView[];
}

/**
 * Rendered ONLY when the backend status says COMPLETED (App gates on it).
 * During validating/replanning this component does not exist in the tree.
 *
 * The destination: completion indicator, validation-derived headline,
 * the synthesized response, and real constraint outcomes.
 */
export function ResultPanel({ status, events }: ResultPanelProps) {
  const validation = useMemo(() => deriveValidation(events), [events]);
  const synthesis = useMemo(() => deriveSynthesis(events), [events]);
  const hasOutcomes =
    validation.satisfied.length > 0 ||
    validation.violated.length > 0 ||
    validation.unresolved.length > 0;

  return (
    <section aria-label="result-panel" data-testid="result-panel" className="band result-band">
      <div className="result-top">
        <span className="chip chip--done">
          <span aria-hidden="true">✓</span> Workflow complete
        </span>
        <span className="panel-meta mono">
          {status.plan_version !== null ? `plan v${status.plan_version} · ` : ''}
          {status.task_count} tasks · {status.observation_count} observations
        </span>
      </div>

      {validation.reason && <h2 className="result-headline">{validation.reason}</h2>}

      <span className="band-label">Synthesized result</span>
      <p data-testid="final-answer" className="result-answer">
        {status.final_answer ?? 'Completed (no final answer recorded.)'}
      </p>

      {hasOutcomes && (
        <ul className="outcomes" aria-label="constraint outcomes">
          {validation.satisfied.map((constraint) => (
            <li key={`satisfied-${constraint}`} className="outcome outcome--pass">
              <span aria-hidden="true">✓</span> {constraint}
              <span className="sr-only"> satisfied</span>
            </li>
          ))}
          {validation.violated.map((constraint) => (
            <li key={`violated-${constraint}`} className="outcome outcome--fail">
              <span aria-hidden="true">!</span> {constraint}
              <span className="sr-only"> violated</span>
            </li>
          ))}
          {validation.unresolved.map((constraint) => (
            <li key={`unresolved-${constraint}`} className="outcome outcome--open">
              <span aria-hidden="true">○</span> {constraint}
              <span className="sr-only"> unresolved</span>
            </li>
          ))}
        </ul>
      )}

      {synthesis && (
        <p className="result-counts mono">
          {synthesis.keyResults !== null && <span>{synthesis.keyResults} key results</span>}
          {synthesis.caveats !== null && <span>{synthesis.caveats} caveats</span>}
          {synthesis.nextActions !== null && (
            <span>{synthesis.nextActions} next actions</span>
          )}
        </p>
      )}

      <p data-testid="result-constraints" className="result-meta mono">
        {status.replan_count} replan{status.replan_count === 1 ? '' : 's'} · plan v
        {status.plan_version ?? '?'} · {status.observation_count} observation
        {status.observation_count === 1 ? '' : 's'}
        {validation.evidence !== null && ` · evidence ${validation.evidence}`}
      </p>
    </section>
  );
}
