import type { ErrorView } from '../api/client';

interface ErrorBannerProps {
  error: ErrorView;
  onDismiss: () => void;
}

export function ErrorBanner({ error, onDismiss }: ErrorBannerProps) {
  const prefix =
    error.kind === 'network'
      ? 'Network error'
      : error.kind === 'api'
        ? `HTTP ${error.status}`
        : 'Invalid input';

  return (
    <div role="alert" aria-label="error-banner" data-testid="error-banner" className="band alert">
      <p className="alert-line">
        <span className="alert-kind mono">{prefix}:</span>{' '}
        <span className="alert-msg">{error.message}</span>
      </p>
      {'kind' in error && error.kind === 'api' && error.issues.length > 0 && (
        <ul aria-label="error-issues" className="alert-issues mono">
          {error.issues.map((issue) => (
            <li key={issue}>{issue}</li>
          ))}
        </ul>
      )}
      <button type="button" onClick={onDismiss} className="btn btn--ghost btn--small">
        Dismiss
      </button>
    </div>
  );
}
