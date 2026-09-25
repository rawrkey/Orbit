import type { FormEvent } from 'react';

import { allAnswered } from '../state/phase';

interface ClarificationFormProps {
  questions: readonly string[];
  answers: Readonly<Record<string, string>>;
  onAnswerChange: (question: string, value: string) => void;
  onSubmit: () => void;
  running: boolean;
}

/**
 * A deliberate interruption in the workflow — the run stays paused until
 * every question has a non-empty answer (allAnswered gates readiness and
 * useOrbit refuses to run while clarifying).
 */
export function ClarificationForm({
  questions,
  answers,
  onAnswerChange,
  onSubmit,
  running,
}: ClarificationFormProps) {
  const ready = allAnswered(questions, answers) && !running;

  function handleSubmit(event: FormEvent): void {
    event.preventDefault();
    if (!ready) return;
    onSubmit();
  }

  return (
    <form aria-label="clarification-form" data-testid="clarification-form" className="band interrupt" onSubmit={handleSubmit}>
      <span className="interrupt-tag mono">Orbit needs one input</span>
      <h2 className="interrupt-title">Workflow paused</h2>
      <p className="interrupt-sub">
        The plan cannot continue until {questions.length === 1 ? 'this question is' : 'these questions are'}{' '}
        answered.
      </p>

      <div className="interrupt-fields">
        {questions.map((question) => (
          <div key={question} className="field">
            <label htmlFor={`answer-${question}`}>{question}</label>
            <input
              id={`answer-${question}`}
              aria-label={question}
              value={answers[question] ?? ''}
              onChange={(event) => onAnswerChange(question, event.target.value)}
            />
          </div>
        ))}
      </div>

      <button type="submit" disabled={!ready} data-testid="clarification-submit" className="btn btn--primary">
        {running ? 'Submitting…' : 'Continue'}
      </button>
    </form>
  );
}
