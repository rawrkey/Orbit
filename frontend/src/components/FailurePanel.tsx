import type { FailureContext } from '../api/types';

/**
 * WHAT failed, WHY the workflow stopped, and WHAT to do next — rendered
 * only when the backend status is FAILED and a failure context exists
 * (App gates on both). The backend status endpoint is already leak-proof
 * (no prompts, keys, or raw observations), and the next-step line points
 * at the only legal recovery: a new goal (failed states cannot resume).
 */
export function FailurePanel({ failure }: { failure: FailureContext }) {
  const scriptExhausted = failure.detail.includes('no scripted');

  return (
    <section aria-label="validation" data-testid="failure-panel" className="band failure-band">
      <div className="panel-head">
        <h2 className="band-label band-label--bad">Workflow halted</h2>
        <span className="panel-meta mono">
          {failure.stage} · {failure.error_code}
        </span>
      </div>
      <p className="failure-reason">
        <strong>Failed at {failure.stage}</strong> — {failure.error_code}
      </p>
      <p className="failure-detail">{failure.detail}</p>
      <p data-testid="failure-next" className="failure-next mono">
        Next: start a new goal — failed workflows cannot be resumed.
      </p>
      {scriptExhausted && (
        <p className="failure-hint mono">
          Offline demo mode: the scripted run is single-use — restart the backend to replay it.
        </p>
      )}
    </section>
  );
}
