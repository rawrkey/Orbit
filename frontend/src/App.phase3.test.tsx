/**
 * Phase 3 — contest demo polish tests: empty state, the demo entry point,
 * adaptive-loop presentation, plan versioning, status/loading language,
 * failure next-steps, keyboard submit, and duplicate-submission safety.
 * Every flow is driven by backend-shaped fixtures — no fabricated events.
 */

import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';

import App from './App';
import eventsDemo from './api/fixtures/events_demo.json';
import statusCompleted from './api/fixtures/status_completed.json';
import statusFailed from './api/fixtures/status_failed.json';
import statusReplanning from './api/fixtures/status_replanning.json';
import { DEMO_GOAL, EXAMPLE_GOALS } from './api/demo';
import { STORAGE_KEY } from './hooks/useOrbit';
import { installFetch, json } from './test/mockFetch';
import { CREATED, VALIDATED_RUN, makeEvent, makePage, makeStatus } from './test/helpers';

const SLOW_MS = 60_000; // effectively disables interval polling in tests

function delayed(data: unknown, ms: number): Promise<Response> {
  return new Promise((resolve) => setTimeout(() => resolve(json(data)), ms));
}

describe('empty state', () => {
  it('says what ORBIT does before anything is typed', () => {
    installFetch(() => json({}, 500));

    render(<App pollMs={SLOW_MS} />);
    expect(
      screen.getByRole('heading', { name: 'Turn a goal into a validated result' }),
    ).toBeInTheDocument();
    expect(screen.getByText(/revises the plan and executes again/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Create workflow' })).toBeInTheDocument();
    expect(screen.getByTestId('demo-button')).toBeInTheDocument();
  });

  it('example chips fill the textarea only — never a network call', async () => {
    const user = userEvent.setup();
    const mock = installFetch(() => json({}, 500));

    render(<App pollMs={SLOW_MS} />);
    const chips = screen.getAllByTestId('example-goal');
    expect(chips).toHaveLength(EXAMPLE_GOALS.length);

    await user.click(chips[0]!);
    expect(screen.getByLabelText('goal-input')).toHaveValue(EXAMPLE_GOALS[0]);
    expect(screen.getByRole('button', { name: 'Create workflow' })).toBeEnabled();

    await user.click(chips[1]!);
    expect(screen.getByLabelText('goal-input')).toHaveValue(EXAMPLE_GOALS[1]);
    expect(mock.calls).toHaveLength(0);
  });
});

describe('demo entry point', () => {
  it('creates the canonical demo goal, runs it, and shows the full adaptive story', async () => {
    const user = userEvent.setup();
    const mock = installFetch((call) => {
      if (call.url === '/workflows' && call.method === 'POST') return json(CREATED, 201);
      if (call.url.endsWith('/run')) return json(VALIDATED_RUN);
      if (call.url.endsWith('/events')) {
        return json(mock.callsTo('/run').length > 0 ? eventsDemo : makePage([makeEvent(0)]));
      }
      if (call.url === '/workflows/wf-test') {
        return json(mock.callsTo('/run').length > 0 ? statusCompleted : makeStatus('RECEIVED'));
      }
      return json({});
    });

    render(<App pollMs={SLOW_MS} />);
    await user.click(screen.getByTestId('demo-button'));

    // completed via backend status; adaptive loop + plan history on screen
    await screen.findByTestId('result-panel');

    // the real API flow used the backend's own demo goal, exactly once,
    // with the EXPLICIT demo flag so the backend serves the scripted run
    const creates = mock.calls.filter(
      (call) => call.url === '/workflows' && call.method === 'POST',
    );
    expect(creates).toHaveLength(1);
    expect(creates[0]?.body).toEqual({ goal: DEMO_GOAL, demo: true });
    expect(mock.callsTo('/run')).toHaveLength(1);

    expect(screen.getByTestId('state-value')).toHaveTextContent('Completed (COMPLETED)');
    expect(screen.getByTestId('final-answer')).toHaveTextContent('7,500');
    expect(screen.getByTestId('adapt-band')).toBeInTheDocument();
    expect(screen.getByTestId('plan-history')).toBeInTheDocument();
    expect(screen.getByTestId('result-attempts')).toHaveTextContent('validated on attempt 2');
    // no run button on a terminal workflow
    expect(screen.queryByTestId('run-button')).not.toBeInTheDocument();
  });

  it('cannot submit twice: double-click yields exactly one create and one run', async () => {
    const mock = installFetch((call) => {
      if (call.url === '/workflows' && call.method === 'POST') return delayed(CREATED, 60);
      if (call.url.endsWith('/run')) return json(VALIDATED_RUN);
      if (call.url.endsWith('/events')) {
        return json(mock.callsTo('/run').length > 0 ? eventsDemo : makePage([makeEvent(0)]));
      }
      if (call.url === '/workflows/wf-test') {
        return json(
          mock.callsTo('/run').length > 0 ? statusCompleted : makeStatus('RECEIVED'),
        );
      }
      return json({});
    });

    render(<App pollMs={SLOW_MS} />);
    fireEvent.click(screen.getByTestId('demo-button'));
    fireEvent.click(screen.getByTestId('demo-button'));

    await screen.findByTestId('result-panel');
    const creates = mock.calls.filter((call) => call.url === '/workflows' && call.method === 'POST');
    expect(creates).toHaveLength(1);
    expect(mock.callsTo('/run')).toHaveLength(1);
  });

  it('disables create + demo controls while a workflow is being created', async () => {
    const user = userEvent.setup();
    const mock = installFetch((call) => {
      if (call.url === '/workflows' && call.method === 'POST') return delayed(CREATED, 80);
      if (call.url.endsWith('/events')) return json(makePage([makeEvent(0)]));
      if (call.url === '/workflows/wf-test') return json(makeStatus('RECEIVED'));
      return json({});
    });

    render(<App pollMs={SLOW_MS} />);
    await user.type(screen.getByLabelText('goal-input'), 'Find options under 8000.');
    await user.click(screen.getByRole('button', { name: 'Create workflow' }));

    expect(await screen.findByRole('button', { name: 'Creating…' })).toBeDisabled();
    expect(screen.getByTestId('demo-button')).toBeDisabled();
    await screen.findByTestId('status-panel');
    expect(mock.calls.filter((call) => call.url === '/workflows')).toHaveLength(1);
  });
});

describe('status header', () => {
  it('shows backend-derived system language while a replan is in progress', async () => {
    window.localStorage.setItem(STORAGE_KEY, 'wf-test');
    installFetch((call) => {
      if (call.url.endsWith('/events')) return json(makePage([makeEvent(0)]));
      if (call.url === '/workflows/wf-test') return json(statusReplanning);
      return json({});
    });

    render(<App pollMs={SLOW_MS} />);
    expect(await screen.findByTestId('system-line')).toHaveTextContent('ADAPTING');
    expect(screen.getByTestId('state-value')).toHaveTextContent('Replanning (REPLANNING)');
    // nothing inferred: no result view, no completion claim
    expect(screen.queryByTestId('result-panel')).not.toBeInTheDocument();
  });

  it('reports the in-flight run with system language', async () => {
    const user = userEvent.setup();
    const mock = installFetch((call) => {
      if (call.url === '/workflows' && call.method === 'POST') return json(CREATED, 201);
      if (call.url.endsWith('/run')) return delayed(VALIDATED_RUN, 100);
      if (call.url.endsWith('/events')) return json(makePage([makeEvent(0)]));
      if (call.url === '/workflows/wf-test') {
        return json(mock.callsTo('/run').length > 0 ? statusCompleted : makeStatus('RECEIVED'));
      }
      return json({});
    });

    render(<App pollMs={SLOW_MS} />);
    await user.type(screen.getByLabelText('goal-input'), 'Find options');
    await user.click(screen.getByRole('button', { name: 'Create workflow' }));
    await user.click(screen.getByTestId('run-button'));

    expect(screen.getByTestId('run-in-progress')).toHaveTextContent(
      'Run in progress — the backend is working…',
    );
    await screen.findByTestId('result-panel');
  });
});

describe('replan presentation', () => {
  it('renders the five-step adaptive ladder from the real event log', async () => {
    window.localStorage.setItem(STORAGE_KEY, 'wf-test');
    installFetch((call) => {
      if (call.url.endsWith('/events')) return json(eventsDemo);
      if (call.url === '/workflows/wf-test') return json(statusCompleted);
      return json({});
    });

    render(<App pollMs={SLOW_MS} />);
    const band = await screen.findByTestId('adapt-band');
    const steps = within(band).getAllByTestId('adapt-step');
    expect(steps).toHaveLength(5);
    expect(steps.map((step) => step.querySelector('.ladder-label')?.textContent)).toEqual([
      'Validation failed',
      'Orbit adapted',
      'Plan revised',
      'Execution resumed',
      'Validation passed',
    ]);
    expect(steps[0]).toHaveTextContent('c1 violated');
    expect(steps[1]).toHaveTextContent('replan 1 of 3');
    expect(steps[2]).toHaveTextContent('v1 → v2');
    expect(steps[4]).toHaveTextContent('c1 satisfied');
    expect(steps.every((step) => step.getAttribute('data-state') === 'done')).toBe(true);
    expect(within(band).getByText('5/5 steps')).toBeInTheDocument();
  });

  it('shows plan versioning: superseded v1 and what the replan changed', async () => {
    window.localStorage.setItem(STORAGE_KEY, 'wf-test');
    installFetch((call) => {
      if (call.url.endsWith('/events')) return json(eventsDemo);
      if (call.url === '/workflows/wf-test') return json(statusCompleted);
      return json({});
    });

    render(<App pollMs={SLOW_MS} />);
    const history = await screen.findByTestId('plan-history');
    expect(history).toHaveTextContent('Plan v1');
    expect(history).toHaveTextContent('superseded');
    expect(history).toHaveTextContent('2/2 tasks completed');
    expect(history).toHaveTextContent('1 constraint failed');
    expect(screen.getByTestId('plan-adapt-link')).toHaveTextContent('v1 → v2');
    expect(screen.getByTestId('plan-adapt-link')).toHaveTextContent('1 preserved');
    expect(screen.getByTestId('plan-adapt-link')).toHaveTextContent('1 revised');
    expect(screen.getByTestId('plan-adapt-link')).toHaveTextContent('0 new');
    // current plan still lists real tasks
    expect(screen.getByText('Recompute cheaper options')).toBeInTheDocument();
  });

  it('stays quiet before any validation failure', async () => {
    const user = userEvent.setup();
    installFetch((call) => {
      if (call.url === '/workflows' && call.method === 'POST') return json(CREATED, 201);
      if (call.url.endsWith('/events')) return json(makePage([makeEvent(0)]));
      if (call.url === '/workflows/wf-test') return json(makeStatus('RECEIVED'));
      return json({});
    });

    render(<App pollMs={SLOW_MS} />);
    await user.type(screen.getByLabelText('goal-input'), 'Find options');
    await user.click(screen.getByRole('button', { name: 'Create workflow' }));
    await screen.findByTestId('status-panel');

    expect(screen.queryByTestId('adapt-band')).not.toBeInTheDocument();
    expect(screen.queryByTestId('plan-history')).not.toBeInTheDocument();
    expect(screen.getByTestId('event-timeline')).toBeInTheDocument();
  });
});

describe('failed state', () => {
  it('communicates what failed and what to do next', async () => {
    window.localStorage.setItem(STORAGE_KEY, 'wf-test');
    installFetch((call) => {
      if (call.url.endsWith('/events')) return json(makePage([makeEvent(0)]));
      if (call.url === '/workflows/wf-test') return json(statusFailed);
      return json({});
    });

    render(<App pollMs={SLOW_MS} />);
    await screen.findByTestId('failure-panel');
    expect(screen.getByTestId('failure-panel')).toHaveTextContent('Workflow halted');
    expect(screen.getByTestId('failure-next')).toHaveTextContent('start a new goal');
    expect(screen.getByTestId('failure-next')).toHaveTextContent('cannot be resumed');
    expect(screen.getByTestId('state-value')).toHaveTextContent('Failed (FAILED)');
    expect(screen.getByRole('button', { name: 'New goal' })).toBeInTheDocument();
    expect(screen.queryByTestId('run-button')).not.toBeInTheDocument();
  });
});

describe('keyboard', () => {
  it('Ctrl+Enter submits the goal from the textarea', async () => {
    const user = userEvent.setup();
    const mock = installFetch((call) => {
      if (call.url === '/workflows' && call.method === 'POST') return json(CREATED, 201);
      if (call.url.endsWith('/events')) return json(makePage([makeEvent(0)]));
      if (call.url === '/workflows/wf-test') return json(makeStatus('RECEIVED'));
      return json({});
    });

    render(<App pollMs={SLOW_MS} />);
    const textarea = screen.getByLabelText('goal-input');
    await user.type(textarea, 'Find options under 8000.');
    fireEvent.keyDown(textarea, { key: 'Enter', ctrlKey: true });

    await screen.findByTestId('status-panel');
    const creates = mock.calls.filter((call) => call.url === '/workflows' && call.method === 'POST');
    expect(creates).toHaveLength(1);
    expect(creates[0]?.body).toEqual({ goal: 'Find options under 8000.' });
  });

  it('plain Enter inside the textarea does not submit', async () => {
    const user = userEvent.setup();
    const mock = installFetch(() => json({}, 500));

    render(<App pollMs={SLOW_MS} />);
    const textarea = screen.getByLabelText('goal-input');
    await user.type(textarea, 'Find options');
    fireEvent.keyDown(textarea, { key: 'Enter' });
    fireEvent.keyDown(textarea, { key: 'Enter', ctrlKey: false });

    await waitFor(() => expect(mock.calls).toHaveLength(0));
    expect(screen.getByLabelText('goal-input')).toBeInTheDocument();
  });
});
