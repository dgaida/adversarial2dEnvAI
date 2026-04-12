"""Example player agent that moves randomly."""

import gymnasium as gym
from .base_agent import Agent


class RandomPlayerAgent(Agent):
    """Example random agent for demonstration."""

    def __init__(self, action_space: gym.spaces.Space):
        self.action_space = action_space

    def get_action(self, observation: dict) -> int:
        """Returns a random action.

        Args:
            observation (dict): The current observation.

        Returns:
            int: A random action from the action space.
        """
        return int(self.action_space.sample())
