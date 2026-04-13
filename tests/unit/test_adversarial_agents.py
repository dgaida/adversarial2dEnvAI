import pytest
import numpy as np
from custom_grid_env.env import CustomGridEnv
from custom_grid_env.agents.adversarial_agents import MinimaxAgent, ExpectimaxAgent


@pytest.fixture
def env():
    return CustomGridEnv()


def test_minimax_agent_init(env):
    agent = MinimaxAgent(env.action_space, env=env)
    assert agent.env == env
    assert agent.depth_limit == 4


def test_expectimax_agent_init(env):
    agent = ExpectimaxAgent(env.action_space, env=env)
    assert agent.env == env
    assert agent.depth_limit == 4


def test_minimax_agent_get_action(env):
    agent = MinimaxAgent(env.action_space, env=env)
    obs = env.reset()[0]
    action = agent.get_action(obs)
    assert action in [0, 1, 2, 3]


def test_expectimax_agent_get_action(env):
    agent = ExpectimaxAgent(env.action_space, env=env)
    obs = env.reset()[0]
    action = agent.get_action(obs)
    assert action in [0, 1, 2, 3]


def test_minimax_ghost_get_action(env):
    agent = MinimaxAgent(env.action_space, env=env)
    # Simulate ghost observation
    obs = {
        "agent_relative_pos": np.array([1, 1]),
        "neighbors": {
            "up": {"accessible": 1, "colour": 0},
            "down": {"accessible": 1, "colour": 0},
            "left": {"accessible": 1, "colour": 0},
            "right": {"accessible": 1, "colour": 0},
        },
    }
    action = agent.get_action(obs)
    assert action in [0, 1, 2, 3]
