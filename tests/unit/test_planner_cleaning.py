import pytest
from unittest.mock import MagicMock, patch
from custom_grid_env.env import CustomGridEnv
from custom_grid_env.planner import TaskPlanner


@pytest.fixture
def planner():
    env = CustomGridEnv()
    # Mock LLMClient to avoid API calls and requirement for API keys
    with patch("custom_grid_env.planner.LLMClient") as mock_llm:
        planner = TaskPlanner(env)
        planner.llm_client = MagicMock()
        yield planner


def test_identify_targets_standard_json(planner):
    planner.llm_client.chat_completion.return_value = "[[2, 2], [0, 4]]"
    targets = planner.identify_targets("Visit piano and dog")
    assert targets == [(2, 2), (0, 4)]


def test_identify_targets_markdown_json(planner):
    planner.llm_client.chat_completion.return_value = "```json\n[[1, 1], [3, 3]]\n```"
    targets = planner.identify_targets("test")
    assert targets == [(1, 1), (3, 3)]


def test_identify_targets_with_think_block(planner):
    # This currently fails based on user report
    planner.llm_client.chat_completion.return_value = (
        "<think>\nI should go to (2,2) and then (0,4)\n</think>\n[[2, 2], [0, 4]]"
    )
    targets = planner.identify_targets("Visit piano and dog")
    # If the current regex is r"\[\s*\[.*\]\s*\]" with re.DOTALL, it MIGHT pick it up if there's no other brackets.
    # But the user says it fails with "Expecting value: line 1 column 1 (char 0)".
    # This happens if json.loads() is called on something that isn't valid JSON.
    assert targets == [(2, 2), (0, 4)]


def test_identify_targets_with_unclosed_think(planner):
    planner.llm_client.chat_completion.return_value = (
        "<think>\nI am thinking...\n[[1, 1]]"
    )
    targets = planner.identify_targets("test")
    assert targets == [(1, 1)]


def test_identify_targets_mixed_text(planner):
    planner.llm_client.chat_completion.return_value = (
        "Here are the coordinates: [[0, 0], [1, 1]] Hope this helps!"
    )
    targets = planner.identify_targets("test")
    assert targets == [(0, 0), (1, 1)]


def test_identify_targets_think_with_brackets(planner):
    # This should fail with current implementation
    planner.llm_client.chat_completion.return_value = (
        "<think>\nVisiting [[2, 2]] might be good.\n</think>\n[[2, 2], [0, 4]]"
    )
    targets = planner.identify_targets("Visit piano and dog")
    assert targets == [(2, 2), (0, 4)]
