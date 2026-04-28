"""Agent that uses Q-Learning to learn the optimal policy."""

import numpy as np
import gymnasium as gym
from typing import Dict, Any, Tuple, Optional
from .base_agent import BaseAgent


class QLearningAgent(BaseAgent):
    """Agent that uses Q-Learning to find the optimal path to the goal.

    Attributes:
        q_table (dict): Dictionary mapping state to action values.
        alpha (float): Learning rate.
        gamma (float): Discount factor.
        epsilon (float): Exploration rate.
    """

    def __init__(
        self,
        action_space: gym.spaces.Space,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.1,
        **kwargs: Any,
    ):
        super().__init__(action_space, **kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}  # Map (row, col) -> np.zeros(action_space.n)

    def _get_q_values(self, state: Tuple[int, int]) -> np.ndarray:
        """Returns Q-values for a given state, initializing if necessary."""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_space.n)
        return self.q_table[state]

    def get_action(self, observation: Dict[str, Any]) -> int:
        """Returns an action based on epsilon-greedy policy.

        Args:
            observation (dict): The current observation.

        Returns:
            int: The action to take.
        """
        # Determine current position
        if self.perceived_agent_pos is not None:
            state = tuple(self.perceived_agent_pos)
        elif self.env is not None:
            state = tuple(self.env.agent_pos)
        else:
            # Fallback for training without env reference
            state = tuple(observation["agent_pos"])

        if np.random.random() < self.epsilon:
            return int(self.action_space.sample())

        q_values = self._get_q_values(state)
        return int(np.argmax(q_values))

    def update(
        self,
        state: Tuple[int, int],
        action: int,
        reward: float,
        next_state: Tuple[int, int],
        done: bool,
    ):
        """Updates the Q-table using the Q-Learning update rule.

        Args:
            state (tuple): Current state (row, col).
            action (int): Action taken.
            reward (float): Reward received.
            next_state (tuple): Next state (row, col).
            done (bool): Whether the episode ended.
        """
        old_q = self._get_q_values(state)[action]

        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self._get_q_values(next_state))

        self.q_table[state][action] = old_q + self.alpha * (target - old_q)

    def get_value(self, state: Tuple[int, int]) -> float:
        """Returns the maximum Q-value for a state.

        Args:
            state (tuple): (row, col) coordinates.

        Returns:
            float: The maximum Q-value.
        """
        return float(np.max(self._get_q_values(state)))

    def get_best_action(self, state: Tuple[int, int]) -> int:
        """Returns the best action for a state according to the Q-table.

        Args:
            state (tuple): (row, col) coordinates.

        Returns:
            int: The best action index.
        """
        return int(np.argmax(self._get_q_values(state)))
