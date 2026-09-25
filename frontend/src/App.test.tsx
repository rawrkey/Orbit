/**
 * End-to-end frontend tests (mocked HTTP, real UI logic).
 *
 * Covers the full flow: goal input -> create -> run -> status/events ->
 * clarification -> trace -> validation/replanning -> synthesis ->
 * COMPLETED, plus all 14 required edge cases. The UI is driven only by
 * backend-shaped responses (fixtures are generated from the live API).
 */

import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';

import App from './App';
import error404 from './api/fixtures/error_404.json';
import error409 from './api/fixtures/error_409.json';
import error422 from './api/fixtures/error_422.json';
import eventsDemo from './api/fixtures/events_demo.json';
import runAwaiting from './api/fixtures/run_awaiting.json';
import runValidated from './api/fixtures/run_validated.json';
import statusAwaiting from './api/fixtures/status_awaiting_clarification.json';
import statusCompleted from './api/fixtures/status_completed.json';
import statusFailed from './api/fixtures/status_failed.json';
import statusReplanning from './api/fixtures/status_replanning.json';
import { WORKFLOW_STATES } from './api/types';
import { STORAGE_KEY } from './hooks/useOrbit';
import { canRun } from './state/phase';
import { installFetch, json, type FetchHandler } from './test/mockFetch';
import {
  CREATED,
  VALIDATED_RUN,
  makeEvent,
  makePage,
  makeStatus,
} from './test/helpers';

const SLOW_MS = 60_000; // effectively disables interval polling in tests

function delayed(data: unknown, ms: number, status = 200): Promise<Response> {
  return new Promise((resolve) => setTimeout(() => resolve(json(data, status)), ms));
}

/** Standard handler: create + status + events for a fresh workflow. */
const freshBackend: FetchHandler = (call) => {
  if (call.url === '/workflows' && call.method === 'POST') return json(CREATED, 201);
  if (call.url.endsWith('/events')) return json(makePage([makeEvent(0)]));
  if (call.url === '/workflows/wf-test') return json(makeStatus('RECEIVED'));
  return json({});
};

async function createWorkflow(user: ReturnType<typeof userEvent.setup>): Promise<void> {
  await user.type(screen.getByLabelText('goal-input'), 'Find options under 8000.');
  await user.click(screen.getByRole('button', { name: 'Create workflow' }));
  await screen.findByTestId('status-panel');
}

describe('full flow', () => {
  it('create -> run -> status/events -> COMPLETED result with multi-pass report', async () => {
    const user = userEvent.setup();
    const mock = installFetch((call) => {
      if (call.url === '/workflows' && call.method === 'POST') return json(CREATED, 201);
      if (call.url.endsWith('/run')) return json(VALIDATED_RUN);
      if (call.url.endsWith('/events?after=0')) return json(eventsDemo);
      if (call.url.endsWith('/events')) {
        // full-page fetch: create sees the first event, the post-run
        // refresh sees the whole story
        return json(mock.callsTo('/run').length > 0 ? eventsDemo : makePage([makeEvent(0)]));
      }
      if (call.url === '/workflows/wf-test') {
        return json(
          mock.calls.some((c) => c.url.endsWith('/run'))
            ? statusCompleted
            : makeStatus('RECEIVED'),
        );
      }
      return json({});
    });

    render(<App pollMs={SLOW_MS} />);
    await createWorkflow(user);

    // Created: backend state RECEIVED, run offered
    expect(screen.getByTestId('state-value')).toHaveTextContent('Received (RECEIVED)');
    expect(screen.getByTestId('run-button')).toHaveTextContent('Run workflow');

    await user.click(screen.getByTestId('run-button'));

    // Terminal only via backend status
    await waitFor(() =>
      expect(screen.getByTestId('state-value')).toHaveTextContent('Completed (COMPLETED)'),
    );
    expect(screen.getByTestId('result-panel')).toBeInTheDocument();
    await waitFor(() => expect(screen.getByTestId('final-answer')).toHaveTextContent('7,500'));

    // Execution trace arrived and is deterministic
    const timeline = within(screen.getByTestId('event-timeline'));
    const items = timeline.getAllByRole('listitem');
    expect(items).toHaveLength(eventsDemo.events.length);
    expect(items[0]).toHaveAttribute('data-seq', '0');
    expect(items[items.length - 1]).toHaveAttribute(
      'data-seq',
      String(eventsDemo.events.length - 1),
    );

    // Multiple execution passes reported (report appears once the run ends)
    await screen.findByTestId('run-report');
    expect(screen.getByTestId('run-report')).toHaveTextContent('validated');
    expect(screen.getByTestId('run-report')).toHaveTextContent('2 execution passes');
    expect(screen.getByTestId('run-report')).toHaveTextContent('1 replan');

    // Terminal states offer no run; exactly one run request total
    expect(screen.queryByTestId('run-button')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'New goal' })).toBeInTheDocument();
    expect(mock.callsTo('/run')).toHaveLength(1);
  });

  it('reports the backend failure without offering a rerun', async () => {
    const user = userEvent.setup();
    let ran = false;
    installFetch((call) => {
      if (call.url === '/workflows' && call.method === 'POST') return json(CREATED, 201);
      if (call.url.endsWith('/run')) {
        ran = true;
        return json(
          {
            workflow_id: 'wf-test',
            success: false,
            final_state: 'FAILED',
            executions: 0,
            validations: 0,
            replans: 0,
            termination: 'analysis_error',
            failure: { stage: 'ANALYZING', error_code: 'LLM_ERROR', detail: 'no model', task_id: null },
          },
          200,
        );
      }
      if (call.url.endsWith('/events')) return json(makePage([makeEvent(0)]));
      if (call.url === '/workflows/wf-test') {
        return json(ran ? statusFailed : makeStatus('RECEIVED'));
      }
      return json({});
    });

    render(<App pollMs={SLOW_MS} />);
    await createWorkflow(user);
    await user.click(screen.getByTestId('run-button'));

    await screen.findByTestId('failure-panel');
    expect(screen.getByTestId('failure-panel')).toHaveTextContent('LLM_ERROR');
    expect(screen.getByTestId('state-value')).toHaveTextContent('Failed (FAILED)');
    expect(screen.queryByTestId('result-panel')).not.toBeInTheDocument();
    expect(screen.queryByTestId('run-button')).not.toBeInTheDocument();
  });
});

describe('edge cases', () => {
  // 1 — empty goal
  it('1. rejects an empty goal client-side without any network call', async () => {
    const user = userEvent.setup();
    const mock = installFetch(() => json({}, 500));

    render(<App pollMs={SLOW_MS} />);
    await user.click(screen.getByRole('button', { name: 'Create workflow' }));

    const banner = await screen.findByTestId('error-banner');
    expect(banner).toHaveTextContent('Goal must not be blank.');
    expect(mock.calls).toHaveLength(0);
    expect(screen.getByLabelText('goal-input')).toBeInTheDocument();
  });

  // 2 — API returns 404
  it('2. a 404 during restore clears the saved workflow and explains', async () => {
    window.localStorage.setItem(STORAGE_KEY, 'wf-test');
    installFetch((call) => {
      if (call.url === '/workflows/wf-test') return json(error404, 404);
      if (call.url.endsWith('/events')) return json(error404, 404);
      return json({});
    });

    render(<App pollMs={SLOW_MS} />);
    expect(await screen.findByTestId('notice')).toHaveTextContent(
      'Saved workflow no longer exists.',
    );
    expect(screen.getByLabelText('goal-input')).toBeInTheDocument();
    expect(screen.queryByTestId('status-panel')).not.toBeInTheDocument();
    expect(window.localStorage.getItem(STORAGE_KEY)).toBeNull();
  });

  // 3 — API returns 409
  it('3. a 409 during clarification is shown and status re-synced, form stays', async () => {
    window.localStorage.setItem(STORAGE_KEY, 'wf-test');
    const user = userEvent.setup();
    const mock = installFetch((call) => {
      if (call.url.endsWith('/run')) return json(error409, 409);
      if (call.url.endsWith('/events')) return json(makePage([makeEvent(0)]));
      if (call.url === '/workflows/wf-test') return json(statusAwaiting);
      return json({});
    });

    render(<App pollMs={SLOW_MS} />);
    await screen.findByTestId('clarification-form');
    await user.type(screen.getByLabelText('Please provide: travel dates'), 'Dec 1-4');
    await user.click(screen.getByTestId('clarification-submit'));

    const banner = await screen.findByTestId('error-banner');
    expect(banner).toHaveTextContent('HTTP 409');
    expect(banner).toHaveTextContent('missing answers');
    // backend status re-synced after the conflict
    await waitFor(() => expect(mock.callsTo('/workflows/wf-test').length).toBeGreaterThanOrEqual(2));
    expect(screen.getByTestId('clarification-form')).toBeInTheDocument();
    expect(screen.getByTestId('state-value')).toHaveTextContent(
      'Awaiting clarification (AWAITING_CLARIFICATION)',
    );
    expect(mock.callsTo('/run')).toHaveLength(1);
  });

  // 4 — API returns 422
  it('4. a 422 surfaces the FastAPI issues list', async () => {
    const user = userEvent.setup();
    installFetch((call) => {
      if (call.url === '/workflows' && call.method === 'POST') return json(error422, 422);
      return json({});
    });

    render(<App pollMs={SLOW_MS} />);
    await user.type(screen.getByLabelText('goal-input'), 'x');
    await user.click(screen.getByRole('button', { name: 'Create workflow' }));

    const banner = await screen.findByTestId('error-banner');
    expect(banner).toHaveTextContent('HTTP 422');
    expect(within(banner).getByLabelText('error-issues')).toHaveTextContent(
      'String should have at least 1 character',
    );
    expect(screen.getByLabelText('goal-input')).toBeInTheDocument();
  });

  // 5 — network request fails
  it('5. network failure shows a network error and a retry succeeds', async () => {
    const user = userEvent.setup();
    const mock = installFetch(() => {
      throw new TypeError('Failed to fetch');
    });

    render(<App pollMs={SLOW_MS} />);
    await user.type(screen.getByLabelText('goal-input'), 'Find options');
    await user.click(screen.getByRole('button', { name: 'Create workflow' }));

    const banner = await screen.findByTestId('error-banner');
    expect(banner).toHaveTextContent('Network error');
    expect(screen.getByLabelText('goal-input')).toBeInTheDocument();

    // retry with a healthy backend
    mock.setHandler(freshBackend);
    await user.click(screen.getByRole('button', { name: 'Create workflow' }));
    await screen.findByTestId('status-panel');
    expect(screen.queryByTestId('error-banner')).not.toBeInTheDocument();
  });

  // 6 — workflow already running (duplicate submissions)
  it('6. a second run submission while one is in flight is impossible', async () => {
    const user = userEvent.setup();
    let runDone = false;
    const mock = installFetch((call) => {
      if (call.url === '/workflows' && call.method === 'POST') return json(CREATED, 201);
      if (call.url.endsWith('/run')) {
        return delayed(VALIDATED_RUN, 80).then((response) => {
          runDone = true;
          return response;
        });
      }
      if (call.url.endsWith('/events')) return json(makePage([makeEvent(0)]));
      if (call.url === '/workflows/wf-test') {
        return json(runDone ? statusCompleted : makeStatus('RECEIVED'));
      }
      return json({});
    });

    render(<App pollMs={SLOW_MS} />);
    await createWorkflow(user);

    const button = screen.getByTestId('run-button');
    await user.click(button);
    // loading state visible + button locked
    expect(screen.getByTestId('run-in-progress')).toBeInTheDocument();
    expect(button).toBeDisabled();
    // hostile: programmatic click while disabled (bypasses pointer checks)
    fireEvent.click(button);

    await waitFor(() => expect(screen.getByTestId('result-panel')).toBeInTheDocument());
    expect(mock.callsTo('/run')).toHaveLength(1);
  });

  // 7 — workflow already completed
  it('7. restoring a COMPLETED workflow shows the result and no run button', async () => {
    window.localStorage.setItem(STORAGE_KEY, 'wf-test');
    installFetch((call) => {
      if (call.url.endsWith('/events')) return json(eventsDemo);
      if (call.url === '/workflows/wf-test') return json(statusCompleted);
      return json({});
    });

    render(<App pollMs={SLOW_MS} />);
    await waitFor(() => expect(screen.getByTestId('result-panel')).toBeInTheDocument());
    expect(screen.getByTestId('state-value')).toHaveTextContent('Completed (COMPLETED)');
    expect(screen.getByTestId('final-answer')).toHaveTextContent('7,500');
    expect(screen.queryByTestId('run-button')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'New goal' })).toBeInTheDocument();
    expect(screen.getByTestId('notice')).toHaveTextContent('Restored workflow');
  });

  // 8 — workflow already failed
  it('8. restoring a FAILED workflow shows the failure and no run button', async () => {
    window.localStorage.setItem(STORAGE_KEY, 'wf-test');
    installFetch((call) => {
      if (call.url.endsWith('/events')) return json(makePage([makeEvent(0)]));
      if (call.url === '/workflows/wf-test') return json(statusFailed);
      return json({});
    });

    render(<App pollMs={SLOW_MS} />);
    await screen.findByTestId('failure-panel');
    expect(screen.getByTestId('failure-panel')).toHaveTextContent('ANALYZING');
    expect(screen.getByTestId('state-value')).toHaveTextContent('Failed (FAILED)');
    expect(screen.queryByTestId('run-button')).not.toBeInTheDocument();
    expect(screen.queryByTestId('result-panel')).not.toBeInTheDocument();
  });

  // 9 — awaiting clarification: no bypass, answers required
  it('9. clarification cannot be bypassed; answers drive the next run', async () => {
    const user = userEvent.setup();
    let runCount = 0;
    const mock = installFetch((call) => {
      if (call.url === '/workflows' && call.method === 'POST') return json(CREATED, 201);
      if (call.url.endsWith('/run')) {
        runCount += 1;
        return runCount === 1 ? json(runAwaiting) : json(VALIDATED_RUN);
      }
      if (call.url.endsWith('/events')) return json(makePage([makeEvent(0)]));
      if (call.url === '/workflows/wf-test') {
        return json(
          runCount === 0
            ? makeStatus('RECEIVED')
            : runCount === 1
              ? statusAwaiting
              : statusCompleted,
        );
      }
      return json({});
    });

    render(<App pollMs={SLOW_MS} />);
    await createWorkflow(user);
    await user.click(screen.getByTestId('run-button'));

    // Backend demands clarification
    await screen.findByTestId('clarification-form');
    expect(screen.getByTestId('state-value')).toHaveTextContent('AWAITING_CLARIFICATION');
    // plain run is impossible while clarifying
    expect(screen.queryByTestId('run-button')).not.toBeInTheDocument();
    // submit blocked while unanswered
    expect(screen.getByTestId('clarification-submit')).toBeDisabled();

    // answering unlocks exactly one run carrying the answers
    await user.type(screen.getByLabelText('Please provide: travel dates'), 'Dec 1-4');
    expect(screen.getByTestId('clarification-submit')).toBeEnabled();
    await user.click(screen.getByTestId('clarification-submit'));

    await waitFor(() => expect(screen.getByTestId('result-panel')).toBeInTheDocument());
    expect(mock.callsTo('/run')).toHaveLength(2);
    const secondRun = mock.callsTo('/run')[1];
    expect(secondRun?.body).toEqual({
      answers: { 'Please provide: travel dates': 'Dec 1-4' },
    });
    expect(screen.getByTestId('state-value')).toHaveTextContent('Completed (COMPLETED)');
  });

  // 10 + 11 — incremental events and empty polls
  it('10/11. polls incrementally with after=, never duplicates, tolerates empty polls', async () => {
    const user = userEvent.setup();
    let runDone = false;
    const mock = installFetch((call) => {
      if (call.url === '/workflows' && call.method === 'POST') return json(CREATED, 201);
      if (call.url.endsWith('/run')) {
        return delayed(VALIDATED_RUN, 150).then((response) => {
          runDone = true;
          return response;
        });
      }
      if (call.url.endsWith('/events?after=0')) return json(makePage([makeEvent(1), makeEvent(2)]));
      if (call.url.endsWith('/events?after=2')) return json(makePage([makeEvent(3)]));
      if (call.url.includes('/events?after=')) return json(makePage([])); // empty polls
      if (call.url.endsWith('/events')) return json(makePage([makeEvent(0)]));
      if (call.url === '/workflows/wf-test') {
        return json(runDone ? statusCompleted : makeStatus('RECEIVED'));
      }
      return json({});
    });

    render(<App pollMs={15} />);
    await createWorkflow(user);
    await user.click(screen.getByTestId('run-button'));
    await waitFor(() => expect(screen.getByTestId('result-panel')).toBeInTheDocument(), {
      timeout: 5000,
    });

    // incremental protocol actually used
    expect(mock.callsTo('events?after=0').length).toBeGreaterThan(0);
    expect(mock.callsTo('events?after=2').length).toBeGreaterThan(0);
    // empty poll happened at least once without breaking anything
    expect(mock.callsTo('events?after=3').length).toBeGreaterThan(0);

    // each event rendered exactly once, in seq order, no duplicates
    const items = within(screen.getByTestId('event-timeline')).getAllByRole('listitem');
    const seqs = items.map((item) => item.getAttribute('data-seq'));
    expect(seqs).toEqual(['0', '1', '2', '3']);
    expect(new Set(seqs).size).toBe(seqs.length);
    expect(mock.callsTo('/run')).toHaveLength(1);
  });

  it('11. polling stops once the backend reports COMPLETED', async () => {
    window.localStorage.setItem(STORAGE_KEY, 'wf-test');
    const mock = installFetch((call) => {
      if (call.url.endsWith('/events')) return json(eventsDemo);
      if (call.url === '/workflows/wf-test') return json(statusCompleted);
      return json({});
    });

    render(<App pollMs={15} />);
    await screen.findByTestId('result-panel');
    const settled = mock.callsTo('/workflows/wf-test').length;
    await new Promise((resolve) => setTimeout(resolve, 120));
    expect(mock.callsTo('/workflows/wf-test')).toHaveLength(settled);
  });

  // 12 — browser refresh during an active workflow
  it('12. refresh mid-run: restores state, resumes event polling, offers resume', async () => {
    window.localStorage.setItem(STORAGE_KEY, 'wf-test');
    const mock = installFetch((call) => {
      if (call.url.endsWith('/events?after=1')) return json(makePage([]));
      if (call.url.endsWith('/events')) {
        return json(
          makePage([
            makeEvent(0, { type: 'workflow.started' }),
            makeEvent(1, { type: 'validation.failed', state: 'VALIDATING' }),
          ]),
        );
      }
      if (call.url === '/workflows/wf-test') return json(statusReplanning);
      return json({});
    });

    render(<App pollMs={15} />);
    await waitFor(() => expect(screen.getByTestId('state-value')).toHaveTextContent('REPLANNING'));
    expect(screen.getByTestId('notice')).toHaveTextContent('Restored workflow');
    expect(screen.getByTestId('run-button')).toHaveTextContent('Resume run');
    expect(screen.queryByTestId('result-panel')).not.toBeInTheDocument();

    // polling resumed: fresh status GETs + incremental event fetches after restore
    await waitFor(
      () => {
        const statusGets = mock.calls.filter((call) => call.url === '/workflows/wf-test').length;
        expect(statusGets).toBeGreaterThan(1);
      },
      { timeout: 5000 },
    );
    await waitFor(() => expect(mock.callsTo('events?after=1').length).toBeGreaterThan(0), {
      timeout: 5000,
    });
  });

  // 13 — validation failure followed by replanning
  it('13. validation failure -> replanning is shown without any final result', async () => {
    window.localStorage.setItem(STORAGE_KEY, 'wf-test');
    installFetch((call) => {
      if (call.url.endsWith('/events')) {
        return json(
          makePage([
            makeEvent(0, { type: 'validation.started', state: 'VALIDATING' }),
            makeEvent(1, { type: 'validation.failed', state: 'VALIDATING' }),
            makeEvent(2, { type: 'replanning.started', state: 'REPLANNING' }),
            makeEvent(3, { type: 'replan.accepted', state: 'REPLANNING' }),
          ]),
        );
      }
      if (call.url === '/workflows/wf-test') return json(statusReplanning);
      return json({});
    });

    render(<App pollMs={SLOW_MS} />);
    await screen.findByTestId('status-panel');
    expect(screen.getByTestId('state-value')).toHaveTextContent('Replanning (REPLANNING)');
    expect(screen.queryByTestId('result-panel')).not.toBeInTheDocument();
    expect(screen.queryByTestId('final-answer')).not.toBeInTheDocument();

    // trace tells the validation-failure story in order
    const types = within(screen.getByTestId('event-timeline'))
      .getAllByRole('listitem')
      .map((item) => item.getAttribute('data-event-type'));
    expect(types).toEqual([
      'validation.started',
      'validation.failed',
      'replanning.started',
      'replan.accepted',
    ]);
  });

  // 14 — completes after multiple execution passes
  it('14. completion after multiple passes: report shows pass/replan counts', async () => {
    const user = userEvent.setup();
    let runDone = false;
    installFetch((call) => {
      if (call.url === '/workflows' && call.method === 'POST') return json(CREATED, 201);
      if (call.url.endsWith('/run')) {
        runDone = true;
        return json(runValidated);
      }
      if (call.url.endsWith('/events')) return json(eventsDemo);
      if (call.url === '/workflows/wf-test') {
        return json(runDone ? statusCompleted : makeStatus('RECEIVED'));
      }
      return json({});
    });

    render(<App pollMs={SLOW_MS} />);
    await createWorkflow(user);
    await user.click(screen.getByTestId('run-button'));

    await screen.findByTestId('run-report');
    expect(screen.getByTestId('run-report')).toHaveTextContent('2 execution passes');
    expect(screen.getByTestId('run-report')).toHaveTextContent('2 validations');
    expect(screen.getByTestId('run-report')).toHaveTextContent('1 replan');
    expect(screen.getByTestId('result-panel')).toBeInTheDocument();
  });
});

describe('backend state invariants', () => {
  it('never displays COMPLETED when the final status refresh fails', async () => {
    const user = userEvent.setup();
    let ran = false;
    installFetch((call) => {
      if (call.url === '/workflows' && call.method === 'POST') return json(CREATED, 201);
      if (call.url.endsWith('/run')) {
        ran = true;
        return json(VALIDATED_RUN); // backend run says COMPLETED...
      }
      if (ran && call.url === '/workflows/wf-test') throw new TypeError('Failed to fetch');
      if (call.url.endsWith('/events')) return json(makePage([makeEvent(0)]));
      if (call.url === '/workflows/wf-test') return json(makeStatus('RECEIVED'));
      return json({});
    });

    render(<App pollMs={SLOW_MS} />);
    await createWorkflow(user);
    await user.click(screen.getByTestId('run-button'));

    await screen.findByTestId('error-banner');
    // status never said COMPLETED -> UI must not claim it
    expect(screen.getByTestId('state-value')).toHaveTextContent('Received (RECEIVED)');
    expect(screen.queryByTestId('result-panel')).not.toBeInTheDocument();
    expect(screen.queryByTestId('final-answer')).not.toBeInTheDocument();
  });

  it.each([
    ['EXECUTING', 'Executing'],
    ['VALIDATING', 'Validating'],
    ['REPLANNING', 'Replanning'],
  ] as const)(
    '%s shows progress without final results',
    async (state, label) => {
      window.localStorage.setItem(STORAGE_KEY, 'wf-test');
      installFetch((call) => {
        if (call.url.endsWith('/events')) return json(makePage([makeEvent(0)]));
        if (call.url === '/workflows/wf-test') return json(makeStatus(state));
        return json({});
      });

      render(<App pollMs={SLOW_MS} />);
      await screen.findByTestId('status-panel');
      expect(screen.getByTestId('state-value')).toHaveTextContent(
        `${label} (${state})`,
      );
      expect(screen.queryByTestId('result-panel')).not.toBeInTheDocument();
      expect(screen.queryByTestId('final-answer')).not.toBeInTheDocument();
      expect(screen.queryByTestId('run-report')).not.toBeInTheDocument();
      window.localStorage.removeItem(STORAGE_KEY);
    },
  );

  it.each(WORKFLOW_STATES)('handles backend state %s', async (state) => {
    window.localStorage.setItem(STORAGE_KEY, 'wf-test');
    installFetch((call) => {
      if (call.url.endsWith('/events')) return json(makePage([]));
      if (call.url === '/workflows/wf-test') {
        return json(
          makeStatus(state, {
            final_answer: state === 'COMPLETED' ? 'Done.' : null,
            clarification_questions:
              state === 'AWAITING_CLARIFICATION' ? ['Please provide: dates'] : [],
          }),
        );
      }
      return json({});
    });

    render(<App pollMs={SLOW_MS} />);
    await waitFor(() =>
      expect(screen.getByTestId('state-value')).toHaveAttribute('data-state', state),
    );

    // run button exists exactly when the backend dispatch would accept it
    if (canRun(state)) {
      expect(screen.getByTestId('run-button')).toBeInTheDocument();
    } else {
      expect(screen.queryByTestId('run-button')).not.toBeInTheDocument();
    }

    // COMPLETED view only for COMPLETED
    if (state === 'COMPLETED') {
      expect(screen.getByTestId('result-panel')).toBeInTheDocument();
    } else {
      expect(screen.queryByTestId('result-panel')).not.toBeInTheDocument();
    }

    // clarification form only while awaiting clarification
    if (state === 'AWAITING_CLARIFICATION') {
      expect(screen.getByTestId('clarification-form')).toBeInTheDocument();
    } else {
      expect(screen.queryByTestId('clarification-form')).not.toBeInTheDocument();
    }

    // failure panel only while failed
    if (state === 'FAILED') {
      // makeStatus has failure: null — panel appears only with a context
      expect(screen.queryByTestId('failure-panel')).not.toBeInTheDocument();
    }

    window.localStorage.removeItem(STORAGE_KEY);
  });
});
