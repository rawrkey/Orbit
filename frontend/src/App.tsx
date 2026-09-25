import { ClarificationForm } from './components/ClarificationForm';
import { ErrorBanner } from './components/ErrorBanner';
import { EventTimeline } from './components/EventTimeline';
import { FailurePanel } from './components/FailurePanel';
import { GoalForm } from './components/GoalForm';
import { PlanPanel } from './components/PlanPanel';
import { ResultPanel } from './components/ResultPanel';
import { StatusPanel } from './components/StatusPanel';
import { useOrbit } from './hooks/useOrbit';
import { isActiveState, isClarifying } from './state/phase';

export interface AppProps {
  /** Polling cadence in ms (tests override). */
  pollMs?: number;
}

export default function App({ pollMs }: AppProps) {
  const orbit = useOrbit({ pollMs });
  const status = orbit.status;

  return (
    <div className="frame">
      <header className="topbar">
        <div className="brand">
          <h1 className="brand-mark">ORBIT</h1>
          <p className="brand-sub mono">adaptive workflow executor</p>
        </div>
        {status && (
          <StatusPanel
            status={status}
            report={orbit.report}
            running={orbit.running}
            onRun={orbit.run}
            onReset={orbit.reset}
          />
        )}
      </header>

      <main className="console">
        {orbit.error && <ErrorBanner error={orbit.error} onDismiss={orbit.dismissError} />}
        {orbit.notice && (
          <div role="status" aria-label="notice" data-testid="notice" className="band notice mono">
            {orbit.notice}
          </div>
        )}

        {!orbit.workflowId && (
          <GoalForm
            goal={orbit.goal}
            creating={orbit.creating}
            onGoalChange={orbit.setGoal}
            onSubmit={orbit.create}
          />
        )}

        {orbit.workflowId && status && (
          <>
            <section aria-label="your goal" className="band goal-band">
              <span className="band-label">Your goal</span>
              <p className="goal-text">{status.goal}</p>
            </section>

            <div className="grid-main">
              <PlanPanel
                events={orbit.events}
                active={orbit.running || isActiveState(status.state)}
              />
              <EventTimeline events={orbit.events} />
            </div>

            {isClarifying(status.state) && (
              <ClarificationForm
                questions={status.clarification_questions}
                answers={orbit.answers}
                onAnswerChange={orbit.setAnswer}
                onSubmit={orbit.run}
                running={orbit.running}
              />
            )}

            {status.state === 'FAILED' && status.failure && (
              <FailurePanel failure={status.failure} />
            )}

            {/* Hard gate: COMPLETED view exists only when the backend says so. */}
            {status.state === 'COMPLETED' && (
              <ResultPanel status={status} events={orbit.events} />
            )}
          </>
        )}
      </main>
    </div>
  );
}
