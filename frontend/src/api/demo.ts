/**
 * Demo + example goal copy.
 *
 * DEMO_GOAL mirrors the backend's own `app/demo.py` DEMO_GOAL exactly: when
 * the backend runs without an OPENROUTER_API_KEY it serves the scripted
 * adaptive story (9000 → FAIL → replan → 7500 → PASS → synthesis) for any
 * goal, and this is the canonical one. EXAMPLE_GOALS are UI suggestions that
 * only fill the textarea — they never submit anything by themselves.
 */

export const DEMO_GOAL = 'Find options under a budget of ₹8,000.';

export const EXAMPLE_GOALS = [
  'Plan a trip within a fixed budget',
  'Compare options under constraints',
  'Analyze a file and produce a summary',
] as const;
