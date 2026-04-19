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
    targets, response = planner.identify_targets("Visit piano and dog")
    assert targets == [(2, 2), (0, 4)]
    assert response == "[[2, 2], [0, 4]]"


def test_identify_targets_markdown_json(planner):
    planner.llm_client.chat_completion.return_value = "```json\n[[1, 1], [3, 3]]\n```"
    targets, response = planner.identify_targets("test")
    assert targets == [(1, 1), (3, 3)]


def test_identify_targets_with_think_block(planner):
    raw_response = (
        "<think>\nI should go to (2,2) and then (0,4)\n</think>\n[[2, 2], [0, 4]]"
    )
    planner.llm_client.chat_completion.return_value = raw_response
    targets, response = planner.identify_targets("Visit piano and dog")
    assert targets == [(2, 2), (0, 4)]
    assert response == raw_response


def test_identify_targets_with_unclosed_think(planner):
    raw_response = "<think>\nI am thinking...\n[[1, 1]]"
    planner.llm_client.chat_completion.return_value = raw_response
    targets, response = planner.identify_targets("test")
    assert targets == [(1, 1)]


def test_identify_targets_mixed_text(planner):
    raw_response = "Here are the coordinates: [[0, 0], [1, 1]] Hope this helps!"
    planner.llm_client.chat_completion.return_value = raw_response
    targets, response = planner.identify_targets("test")
    assert targets == [(0, 0), (1, 1)]


def test_identify_targets_think_with_brackets(planner):
    raw_response = (
        "<think>\nVisiting [[2, 2]] might be good.\n</think>\n[[2, 2], [0, 4]]"
    )
    planner.llm_client.chat_completion.return_value = raw_response
    targets, response = planner.identify_targets("Visit piano and dog")
    assert targets == [(2, 2), (0, 4)]


def test_identify_targets_user_truncated_response(planner):
    # This is the exact response provided by the user that failed
    failed_response = (
        "<think>\n"
        "Okay, let's see. The user wants me to find the optimal path visiting three specific fields and then return to the starting point. The starting point is mentioned as the field with the text 'Start', which is at (0,2). \n\n"
        "First, I need to identify the three target fields. Let me check each one:\n\n"
        "1. The field where you can hear piano music. Looking at the grid, field (2,2) has classical and piano music. So that's (2,2).\n\n"
        "2. The field with the dog picture and rock music. The description says (0,4) has a dog image and rock music. So that's (0,4).\n\n"
        "3. The field with the 'Ziel' text. That's at (3,1). \n\n"
        "Now, the task is to visit these three fields in optimal order and return to the start. The start is (0,2). Optimal path usually means the shortest route, minimizing steps. Let me think about the order.\n\n"
        "Starting from (0,2), the first target could be either (2,2), (0,4), or (3,1). Let's see the distances:\n\n"
        "From (0,2) to (2,2): That's 2 steps down (rows increase by 2), same column. So 2 steps.\n\n"
        "From (0,2) to (0,4): Moving right 2 columns. So 2 steps.\n\n"
        "From (0,2) to (3,1): Moving down 3 rows and left 1 column. Total steps: 3 + 1 = 4.\n\n"
        "So maybe visiting (0,4) first is shorter. Then from (0,4) to (2,2) or (3,1). Let's see:\n\n"
        "From (0,4) to (2,2): Down 2 rows and left 2 columns. Total steps: 4.\n\n"
        "From (0,4) to (3,1): Down 3 rows and left 3 columns. Steps: 6.\n\n"
        "Alternatively, visiting (2,2) first. From (0,2) to (2,2) is 2 steps. Then from (2,2) to (0,4): Up 2 rows and right 2 columns. Steps: 4. Then from (0,4) to (3,1): Down 3 rows and left 3 columns. Steps: 6. Then return to (0"
    )
    planner.llm_client.chat_completion.return_value = failed_response
    targets, response = planner.identify_targets("some task")
    # It should return an empty list because no JSON array [[...]] was found
    assert targets == []
