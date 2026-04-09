import gymnasium as gym
from custom_grid_env.agents.chase_ghost_agent import ChaseGhostAgent
from custom_grid_env.agents.random_ghost_agent import RandomGhostAgent
from custom_grid_env.agents.random_player_agent import RandomPlayerAgent


def test_chase_ghost_agent():
    action_space = gym.spaces.Discrete(4)
    agent = ChaseGhostAgent(action_space)

    # Mock observation: agent is below ghost
    obs = {
        "agent_relative_pos": [2, 0],
        "neighbors": {
            "down": {"accessible": 1, "colour": 0},
            "up": {"accessible": 1, "colour": 0},
            "right": {"accessible": 1, "colour": 0},
            "left": {"accessible": 1, "colour": 0},
        },
    }
    action = agent.get_action(obs)
    assert action == 1  # Should move DOWN


def test_random_ghost_agent():
    action_space = gym.spaces.Discrete(4)
    agent = RandomGhostAgent(action_space)
    obs = {}
    action = agent.get_action(obs)
    assert action in [0, 1, 2, 3]


def test_random_player_agent():
    action_space = gym.spaces.Discrete(4)
    agent = RandomPlayerAgent(action_space)
    obs = {}
    action = agent.get_action(obs)
    assert action in [0, 1, 2, 3]
