import gymnasium as gym
import numpy as np
from custom_grid_env.agents.q_learning_agent import QLearningAgent


def test_q_learning_init():
    action_space = gym.spaces.Discrete(4)
    agent = QLearningAgent(action_space)
    assert agent.alpha == 0.1
    assert agent.gamma == 0.9
    assert agent.epsilon == 0.1
    assert isinstance(agent.q_table, dict)


def test_q_learning_update():
    action_space = gym.spaces.Discrete(4)
    agent = QLearningAgent(
        action_space, alpha=1.0, gamma=0.9
    )  # alpha=1.0 to see full change

    state = (0, 0)
    action = 1
    reward = 10.0
    next_state = (0, 1)

    # Pre-set some value for next state
    agent.q_table[next_state] = np.array([0, 0, 100, 0])  # Best next Q is 100

    agent.update(state, action, reward, next_state, False)

    # Q(s,a) = 0 + 1.0 * (10 + 0.9 * 100 - 0) = 100
    assert agent.q_table[state][action] == 100.0


def test_q_learning_get_value():
    action_space = gym.spaces.Discrete(4)
    agent = QLearningAgent(action_space)
    state = (1, 1)
    agent.q_table[state] = np.array([5, 10, 2, 8])
    assert agent.get_value(state) == 10.0
    assert agent.get_best_action(state) == 1


def test_q_learning_action():
    action_space = gym.spaces.Discrete(4)
    agent = QLearningAgent(action_space, epsilon=0.0)  # Deterministic
    state = (2, 2)
    agent.q_table[state] = np.array([0, 50, 0, 0])

    obs = {"agent_pos": [2, 2]}
    assert agent.get_action(obs) == 1
