import { vi } from 'vitest';

export interface RecordedCall {
  url: string;
  method: string;
  body: unknown;
}

export type FetchHandler = (call: RecordedCall) => Response | Promise<Response>;

export interface FetchMock {
  calls: RecordedCall[];
  setHandler: (handler: FetchHandler) => void;
  /** Calls matching a path fragment, in order. */
  callsTo: (fragment: string) => RecordedCall[];
}

/**
 * Installs a scripted fetch. The handler is swappable mid-test so flows can
 * change phase (create -> run -> status refresh) without remounting.
 */
export function installFetch(initial: FetchHandler): FetchMock {
  let handler = initial;
  const calls: RecordedCall[] = [];

  const mock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const url =
      typeof input === 'string'
        ? input
        : input instanceof URL
          ? input.href
          : input.url;
    const method = (init?.method ?? 'GET').toUpperCase();
    let body: unknown;
    if (typeof init?.body === 'string') {
      try {
        body = JSON.parse(init.body) as unknown;
      } catch {
        body = init.body;
      }
    }
    const call: RecordedCall = { url, method, body };
    calls.push(call);
    return handler(call);
  });

  vi.stubGlobal('fetch', mock);

  return {
    calls,
    setHandler: (next: FetchHandler) => {
      handler = next;
    },
    callsTo: (fragment: string) => calls.filter((call) => call.url.includes(fragment)),
  };
}

export function json(data: unknown, status = 200): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

export function networkFailure(): never {
  throw new TypeError('Failed to fetch');
}
