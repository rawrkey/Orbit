import type { FormEvent, KeyboardEvent } from 'react';

import { EXAMPLE_GOALS } from '../api/demo';

interface GoalFormProps {
  goal: string;
  creating: boolean;
  onGoalChange: (value: string) => void;
  onSubmit: () => void;
  onDemo: () => void;
}

/**
 * First impression: what ORBIT does, a textarea, and two ways in — your own
 * goal or the scripted demo. Example chips only fill the textarea; they never
 * submit. Ctrl/Cmd+Enter submits from the keyboard.
 */
export function GoalForm({ goal, creating, onGoalChange, onSubmit, onDemo }: GoalFormProps) {
  function handleSubmit(event: FormEvent): void {
    event.preventDefault();
    onSubmit();
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>): void {
    if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) {
      event.preventDefault();
      onSubmit();
    }
  }

  return (
    <form aria-label="goal-form" className="band intake" onSubmit={handleSubmit}>
      <span className="band-label">New workflow</span>
      <h2 className="intake-title">Turn a goal into a validated result</h2>
      <p className="intake-sub">
        Plan → act → observe → validate → adapt. When a constraint fails, ORBIT revises the plan
        and executes again — every step visible in the execution trace.
      </p>

      <label htmlFor="goal-input">Your goal</label>
      <textarea
        id="goal-input"
        aria-label="goal-input"
        value={goal}
        onChange={(event) => onGoalChange(event.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Describe what you want done..."
        rows={4}
      />

      <div className="intake-actions">
        <button type="submit" disabled={creating} className="btn btn--primary">
          {creating ? 'Creating…' : 'Create workflow'}
        </button>
        <button
          type="button"
          data-testid="demo-button"
          disabled={creating}
          onClick={onDemo}
          className="btn btn--demo"
        >
          Try demo
        </button>
      </div>

      <div className="examples">
        <span className="examples-label mono">Try an example</span>
        {EXAMPLE_GOALS.map((example) => (
          <button
            key={example}
            type="button"
            data-testid="example-goal"
            className="example"
            onClick={() => onGoalChange(example)}
          >
            {example}
          </button>
        ))}
      </div>
    </form>
  );
}
