"""State machine tests: transition table coverage, illegal transitions,
terminal protection, and FAILED reachability."""

from __future__ import annotations

import pytest

from app.agent.state_machine import (
    LEGAL_TRANSITIONS,
    IllegalTransitionError,
    assert_legal,
    is_legal,
    is_terminal,
    legal_targets,
)
from app.state.models import WorkflowState

ALL_STATES = list(WorkflowState)
S = WorkflowState


def test_table_covers_every_state() -> None:
    assert set(LEGAL_TRANSITIONS) == set(ALL_STATES)


def test_table_targets_are_valid_states() -> None:
    for source, targets in LEGAL_TRANSITIONS.items():
        assert targets <= set(ALL_STATES), f"bad targets for {source}"


def test_happy_path_is_fully_legal() -> None:
    happy_path = [
        S.RECEIVED,
        S.ANALYZING,
        S.PLANNING,
        S.READY,
        S.EXECUTING,
        S.OBSERVING,
        S.VALIDATING,
        S.EXECUTING,
        S.OBSERVING,
        S.VALIDATING,
        S.COMPLETED,
    ]
    for source, target in zip(happy_path, happy_path[1:]):
        assert is_legal(source, target), f"{source} -> {target} should be legal"


def test_replanning_path_is_legal() -> None:
    replan_path = [S.VALIDATING, S.REPLANNING, S.PLANNING, S.READY, S.EXECUTING]
    for source, target in zip(replan_path, replan_path[1:]):
        assert is_legal(source, target), f"{source} -> {target} should be legal"


@pytest.mark.parametrize(
    ("source", "target"),
    [
        (S.RECEIVED, S.EXECUTING),       # skip analysis/planning
        (S.RECEIVED, S.COMPLETED),       # skip everything
        (S.ANALYZING, S.VALIDATING),     # skip planning/execution
        (S.PLANNING, S.EXECUTING),       # skip READY
        (S.READY, S.VALIDATING),         # skip execution
        (S.EXECUTING, S.REPLANNING),     # must go through OBSERVING/VALIDATING
        (S.EXECUTING, S.COMPLETED),      # must validate first
        (S.OBSERVING, S.PLANNING),       # must validate first
        (S.VALIDATING, S.ANALYZING),     # no going back to analysis
        (S.REPLANNING, S.EXECUTING),     # replan goes through PLANNING/READY
        (S.RECEIVED, S.RECEIVED),        # self-transition
    ],
)
def test_illegal_transitions_are_rejected(
    source: WorkflowState, target: WorkflowState
) -> None:
    assert not is_legal(source, target)
    with pytest.raises(IllegalTransitionError) as exc_info:
        assert_legal(source, target)
    assert exc_info.value.source == source
    assert exc_info.value.target == target
    assert source.value in str(exc_info.value)
    assert target.value in str(exc_info.value)


@pytest.mark.parametrize("state", [S.COMPLETED, S.FAILED])
def test_terminal_states_have_no_outgoing_transitions(state: WorkflowState) -> None:
    assert is_terminal(state)
    assert legal_targets(state) == frozenset()
    for target in ALL_STATES:
        assert not is_legal(state, target)
        with pytest.raises(IllegalTransitionError):
            assert_legal(state, target)


@pytest.mark.parametrize(
    "source", [s for s in ALL_STATES if s not in (S.COMPLETED, S.FAILED)]
)
def test_failed_is_reachable_from_every_non_terminal_state(
    source: WorkflowState,
) -> None:
    assert is_legal(source, S.FAILED)


def test_completed_only_reachable_from_validating() -> None:
    sources = [s for s in ALL_STATES if is_legal(s, S.COMPLETED)]
    assert sources == [S.VALIDATING]


def test_error_message_includes_reason_when_given() -> None:
    with pytest.raises(IllegalTransitionError, match="because"):
        assert_legal(S.RECEIVED, S.COMPLETED, reason="because")
