/**
 * OrbitClient unit tests: URLs, bodies, and the 404/409/422/network error
 * contract — checked against real backend-generated fixtures.
 */

import { describe, expect, it } from 'vitest';

import error404 from './fixtures/error_404.json';
import error409 from './fixtures/error_409.json';
import error422 from './fixtures/error_422.json';
import { ApiError, NetworkError, OrbitClient } from './client';
import { installFetch, json } from '../test/mockFetch';

const client = new OrbitClient();

describe('OrbitClient requests', () => {
  it('POSTs /workflows with the goal and parses the 201 body', async () => {
    const mock = installFetch(() => json({ workflow_id: 'wf1', state: 'RECEIVED' }, 201));
    const result = await client.createWorkflow('Find options under 8000.');
    expect(result).toEqual({ workflow_id: 'wf1', state: 'RECEIVED' });
    expect(mock.calls).toHaveLength(1);
    expect(mock.calls[0]?.url).toBe('/workflows');
    expect(mock.calls[0]?.method).toBe('POST');
    // normal creation sends exactly the goal - no demo flag
    expect(mock.calls[0]?.body).toEqual({ goal: 'Find options under 8000.' });
  });

  it('POSTs /workflows with demo: true only when explicitly requested', async () => {
    const mock = installFetch(() => json({ workflow_id: 'wf1', state: 'RECEIVED' }, 201));
    await client.createWorkflow('Find options under a budget of ₹8,000.', {
      demo: true,
    });
    expect(mock.calls[0]?.body).toEqual({
      goal: 'Find options under a budget of ₹8,000.',
      demo: true,
    });

    // demo: false never rides along on the wire
    const mock2 = installFetch(() => json({ workflow_id: 'wf2', state: 'RECEIVED' }, 201));
    await client.createWorkflow('Find options under 8000.', { demo: false });
    expect(mock2.calls[0]?.body).toEqual({ goal: 'Find options under 8000.' });
  });

  it('GETs /workflows/{id}', async () => {
    const mock = installFetch(() => json({ workflow_id: 'wf1' }));
    await client.getWorkflow('wf1');
    expect(mock.calls[0]?.url).toBe('/workflows/wf1');
    expect(mock.calls[0]?.method).toBe('GET');
  });

  it('POSTs /workflows/{id}/run with answers (or empty object)', async () => {
    const mock = installFetch(() => json({}));
    await client.runWorkflow('wf1', { 'Please provide: dates': 'Dec 1-4' });
    expect(mock.calls[0]?.url).toBe('/workflows/wf1/run');
    expect(mock.calls[0]?.method).toBe('POST');
    expect(mock.calls[0]?.body).toEqual({
      answers: { 'Please provide: dates': 'Dec 1-4' },
    });

    const mock2 = installFetch(() => json({}));
    await client.runWorkflow('wf1');
    expect(mock2.calls[0]?.body).toEqual({ answers: {} });
  });

  it('GETs events with after= only when provided', async () => {
    const mock = installFetch(() => json({ workflow_id: 'wf1', events: [], count: 0 }));
    await client.getEvents('wf1');
    expect(mock.calls[0]?.url).toBe('/workflows/wf1/events');
    await client.getEvents('wf1', 7);
    expect(mock.calls[1]?.url).toBe('/workflows/wf1/events?after=7');
  });
});

describe('OrbitClient error mapping', () => {
  it('maps 404 to ApiError with the backend detail (fixture)', async () => {
    installFetch(() => json(error404, 404));
    await expect(client.getWorkflow('missing')).rejects.toMatchObject({
      name: 'ApiError',
      status: 404,
    });
    try {
      await client.getWorkflow('missing');
    } catch (error) {
      expect(error).toBeInstanceOf(ApiError);
      expect((error as ApiError).message).toContain('not found');
    }
  });

  it('maps 409 to ApiError carrying the controller rejection (fixture)', async () => {
    installFetch(() => json(error409, 409));
    try {
      await client.runWorkflow('wf1', { wrong: 'x' });
      expect.unreachable('should have thrown');
    } catch (error) {
      expect(error).toBeInstanceOf(ApiError);
      expect((error as ApiError).status).toBe(409);
      expect((error as ApiError).message).toContain('missing answers');
    }
  });

  it('maps 422 to ApiError with parsed FastAPI issues (fixture)', async () => {
    installFetch(() => json(error422, 422));
    try {
      await client.createWorkflow('');
      expect.unreachable('should have thrown');
    } catch (error) {
      expect(error).toBeInstanceOf(ApiError);
      const apiError = error as ApiError;
      expect(apiError.status).toBe(422);
      expect(apiError.issues.length).toBeGreaterThan(0);
      expect(apiError.issues[0]).toContain('at least 1 character');
    }
  });

  it('maps fetch rejection to NetworkError', async () => {
    installFetch(() => {
      throw new TypeError('Failed to fetch');
    });
    await expect(client.getWorkflow('wf1')).rejects.toBeInstanceOf(NetworkError);
  });

  it('maps a non-JSON error body to a generic ApiError', async () => {
    installFetch(() => new Response('boom', { status: 500 }));
    try {
      await client.getWorkflow('wf1');
      expect.unreachable('should have thrown');
    } catch (error) {
      expect(error).toBeInstanceOf(ApiError);
      expect((error as ApiError).status).toBe(500);
    }
  });
});
