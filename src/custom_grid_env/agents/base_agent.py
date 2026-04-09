from abc import ABC, abstractmethod
import gymnasium as gym


class Agent(ABC):
    """Base class for all agents in the custom grid environment.

    Attributes:
        action_space (gym.spaces.Space): The action space of the environment.
    """

    def __init__(self, action_space: gym.spaces.Space):
        """Initializes the agent with the given action space.

        Args:
            action_space (gym.spaces.Space): The action space of the environment.
        """
        self.action_space = action_space

    @abstractmethod
    def get_action(self, observation: dict) -> int:
        """Returns an action based on the given observation.

        Args:
            observation (dict): The current observation from the environment.

        Returns:
            int: The action to take.
        """
        pass
