import { memo } from 'react';

import type { EventView } from '../api/types';
import { clockTime, eventCategory } from '../state/eventVisual';

interface EventTimelineProps {
  events: readonly EventView[];
}

/**
 * The identity of ORBIT: a dense, deterministic execution trace.
 *
 * Rendering is memoized per event (keyed by backend seq) so status-only
 * polls do not re-render the whole list — only genuinely new rows mount,
 * and the entrance animation runs once per event.
 */
const TraceRow = memo(function TraceRow({ event }: { event: EventView }) {
  const category = eventCategory(event.type);
  return (
    <li
      data-seq={event.seq}
      data-event-type={event.type}
      data-cat={category}
      className="ev"
    >
      <span className="ev-time mono">{clockTime(event.timestamp)}</span>
      <span className="ev-cat mono">{category}</span>
      <code className="ev-type">{event.type}</code>
      <span className="ev-state mono">{event.state}</span>
      {event.task_id && <code className="ev-task mono">{event.task_id}</code>}
      {event.decision_summary && <span className="ev-summary">{event.decision_summary}</span>}
    </li>
  );
});

export function EventTimeline({ events }: EventTimelineProps) {
  return (
    <section aria-label="event-timeline" data-testid="event-timeline" className="band trace-panel">
      <div className="panel-head">
        <h2 className="band-label">Execution trace</h2>
        <span className="panel-meta mono">
          {events.length} event{events.length === 1 ? '' : 's'}
        </span>
      </div>
      {events.length > 0 ? (
        <ol className="trace">
          {events.map((event) => (
            <TraceRow key={event.seq} event={event} />
          ))}
        </ol>
      ) : (
        <p className="empty mono">Waiting for the first event…</p>
      )}
    </section>
  );
}
