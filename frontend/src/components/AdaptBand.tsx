import { useMemo } from 'react';
import type { CSSProperties } from 'react';

import type { EventView } from '../api/types';
import { deriveReplanChain, type ChainStep } from '../state/eventVisual';

interface AdaptBandProps {
  events: readonly EventView[];
}

const TONE_LABEL: Record<ChainStep['tone'], string> = {
  fail: 'failure',
  adapt: 'adaptation',
  pass: 'success',
};

/**
 * The adaptive loop made unmissable: validation failed → orbit adapted →
 * plan revised → execution resumed → validation passed. Renders only when
 * the real event log contains a validation failure or a replan; steps
 * advance as their source events arrive (staggered, reduced-motion aware).
 */
export function AdaptBand({ events }: AdaptBandProps) {
  const steps = useMemo(() => deriveReplanChain(events), [events]);
  if (!steps) return null;
  const done = steps.filter((step) => step.state === 'done').length;

  return (
    <section aria-label="adaptive loop" data-testid="adapt-band" className="band adapt-band">
      <div className="panel-head">
        <h2 className="band-label band-label--accent">Adaptive loop</h2>
        <span className="panel-meta mono">
          {done}/{steps.length} steps
        </span>
      </div>
      <ol className="ladder">
        {steps.map((step, index) => (
          <li
            key={step.label}
            data-state={step.state}
            data-tone={step.tone}
            data-testid="adapt-step"
            className={`ladder-step ladder-step--${step.state}`}
            style={{ '--i': index } as CSSProperties}
          >
            <span className="ladder-label">{step.label}</span>
            {step.detail && <span className="ladder-detail mono">{step.detail}</span>}
            <span className="sr-only">
              {' '}
              — {step.state === 'done' ? 'complete' : step.state}, {TONE_LABEL[step.tone]}
            </span>
          </li>
        ))}
      </ol>
    </section>
  );
}
