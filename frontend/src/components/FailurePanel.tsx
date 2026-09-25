import type { FailureContext } from '../api/types';

/**
 * The VALIDATION band: shown only when the backend status is FAILED and a
 * failure context exists (App gates on both — see the state invariants).
 */
export function FailurePanel({ failure }: { failure: FailureContext }) {
  return (
    <section aria-label="validation" data-testid="failure-panel" className="band failure-band">
      <div className="panel-head">
        <h2 className="band-label band-label--bad">Validation halted</h2>
        <span className="panel-meta mono">
          {failure.stage} · {failure.error_code}
        </span>
      </div>
      <p className="failure-reason">
        <strong>Failed at {failure.stage}</strong> — {failure.error_code}
      </p>
      <p className="failure-detail">{failure.detail}</p>
    </section>
  );
}
