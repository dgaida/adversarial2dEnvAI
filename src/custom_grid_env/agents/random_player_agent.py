"""Example player agent that moves randomly."""

from .base_agent import BaseAgent


class RandomPlayerAgent(BaseAgent):
    """Example random agent for demonstration."""

    def get_action(self, observation: dict) -> int:
        """Returns a random action.

        Args:
            observation (dict): The current observation.

        Returns:
            int: A random action from the action space.
        """
        return int(self.action_space.sample())
