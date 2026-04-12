from typing import Protocol, runtime_checkable, Any, Dict
import gymnasium as gym


@runtime_checkable
class Agent(Protocol):
    """Protocol for all agents in the custom grid environment.

    Any class that implements a 'get_action' method with the correct signature
    is considered an Agent.
    """

    def __init__(self, action_space: gym.spaces.Space):
        """Initializes the agent with the given action space."""
        ...

    def get_action(self, observation: Dict[str, Any]) -> int:
        """Returns an action based on the given observation.

        Args:
            observation (dict): The current observation from the environment.

        Returns:
            int: The action to take.
        """
        ...
