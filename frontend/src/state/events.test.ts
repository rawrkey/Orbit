/**
 * Event accumulation: deterministic order, seq de-duplication, empty-poll
 * identity (no re-render), and incremental lastSeq advancement.
 */

import { describe, expect, it } from 'vitest';

import { parseEvents } from '../api/contract';
import eventsDemo from '../api/fixtures/events_demo.json';
import { maxSeq, mergeEvents } from './events';
import { makeEvent } from '../test/helpers';

describe('mergeEvents', () => {
  it('merges and sorts by seq regardless of arrival order', () => {
    const existing = [makeEvent(0), makeEvent(1)];
    const incoming = [makeEvent(3), makeEvent(2)];
    const result = mergeEvents(existing, incoming, 1);
    expect(result.events.map((event) => event.seq)).toEqual([0, 1, 2, 3]);
    expect(result.lastSeq).toBe(3);
  });

  it('never duplicates events when after= overlaps (fixture page re-delivered)', () => {
    const existing = [makeEvent(0), makeEvent(1)];
    // Backend re-sends the whole page (as if after= were ignored)
    const incoming = [makeEvent(0), makeEvent(1), makeEvent(2)];
    const result = mergeEvents(existing, incoming, 1);
    expect(result.events.map((event) => event.seq)).toEqual([0, 1, 2]);
    expect(new Set(result.events.map((event) => event.seq)).size).toBe(3);
  });

  it('drops stale incoming events with seq <= lastSeq', () => {
    const existing = [makeEvent(0), makeEvent(1), makeEvent(2)];
    const incoming = [makeEvent(1), makeEvent(0)];
    const result = mergeEvents(existing, incoming, 2);
    expect(result.events).toBe(existing); // same reference: no re-render
    expect(result.lastSeq).toBe(2);
  });

  it('returns the SAME array reference when nothing is new (empty poll)', () => {
    const existing = [makeEvent(0)];
    const result = mergeEvents(existing, [], 0);
    expect(result.events).toBe(existing);
    expect(result.lastSeq).toBe(0);
  });

  it('de-duplicates duplicate seqs inside one incoming page', () => {
    const incoming = [makeEvent(5), makeEvent(5), makeEvent(6)];
    const result = mergeEvents([], incoming, -1);
    expect(result.events.map((event) => event.seq)).toEqual([5, 6]);
  });

  it('keeps payload/summary of the delivered copy for a fresh seq', () => {
    const stale = makeEvent(1, { decision_summary: 'old' });
    const fresh = makeEvent(1, { decision_summary: 'new' });
    const result = mergeEvents([stale], [fresh], -1);
    expect(result.events[0]?.decision_summary).toBe('new');
  });
});

describe('maxSeq', () => {
  it('returns -1 for an empty page and the max seq otherwise', () => {
    expect(maxSeq([])).toBe(-1);
    expect(maxSeq([makeEvent(0), makeEvent(9), makeEvent(4)])).toBe(9);
  });

  it('works with real fixture events', () => {
    const fixture = parseEvents(eventsDemo);
    expect(maxSeq(fixture.events)).toBe(fixture.events.length - 1);
  });
});
