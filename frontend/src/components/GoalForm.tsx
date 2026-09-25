import type { FormEvent } from 'react';

interface GoalFormProps {
  goal: string;
  creating: boolean;
  onGoalChange: (value: string) => void;
  onSubmit: () => void;
}

export function GoalForm({ goal, creating, onGoalChange, onSubmit }: GoalFormProps) {
  function handleSubmit(event: FormEvent): void {
    event.preventDefault();
    onSubmit();
  }

  return (
    <form aria-label="goal-form" className="band intake" onSubmit={handleSubmit}>
      <span className="band-label">New workflow</span>
      <h2 className="intake-title">Give ORBIT a goal</h2>
      <p className="intake-sub">
        It will plan, act, observe, validate, and adapt — every step visible in the execution
        trace.
      </p>

      <label htmlFor="goal-input">Your goal</label>
      <textarea
        id="goal-input"
        aria-label="goal-input"
        value={goal}
        onChange={(event) => onGoalChange(event.target.value)}
        placeholder="Describe what you want to accomplish"
        rows={4}
      />
      <button type="submit" disabled={creating} className="btn btn--primary">
        {creating ? 'Creating…' : 'Create workflow'}
      </button>
    </form>
  );
}
