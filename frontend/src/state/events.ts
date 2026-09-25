/**
 * Deterministic event accumulation.
 *
 * - Events are keyed and ordered by their backend-assigned `seq` (append-only
 *   on the backend, so seq is globally unique per workflow).
 * - Polling with `after=<lastSeq>` can never duplicate an event: incoming
 *   seqs <= lastSeq are dropped, overlapping seqs are de-duplicated by Map.
 * - When a poll yields nothing new the EXISTING array reference is returned,
 *   so React skips the re-render entirely.
 */

import type { EventView } from '../api/types';

export interface EventAccumulation {
  events: EventView[];
  lastSeq: number;
}

export function mergeEvents(
  existing: readonly EventView[],
  incoming: readonly EventView[],
  lastSeq: number,
): EventAccumulation {
  const fresh = incoming.filter((event) => event.seq > lastSeq);
  if (fresh.length === 0) {
    return { events: existing as EventView[], lastSeq };
  }

  const bySeq = new Map<number, EventView>();
  for (const event of existing) bySeq.set(event.seq, event);
  for (const event of fresh) bySeq.set(event.seq, event);

  const events = Array.from(bySeq.values()).sort((a, b) => a.seq - b.seq);
  let max = lastSeq;
  for (const event of fresh) {
    if (event.seq > max) max = event.seq;
  }
  return { events, lastSeq: max };
}

/** Highest seq in a page (or -1 when the page is empty). */
export function maxSeq(events: readonly EventView[]): number {
  let max = -1;
  for (const event of events) {
    if (event.seq > max) max = event.seq;
  }
  return max;
}
