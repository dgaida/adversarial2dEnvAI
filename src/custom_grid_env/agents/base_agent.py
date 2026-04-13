from typing import Protocol, runtime_checkable, Any, Dict
import gymnasium as gym


@runtime_checkable
class Agent(Protocol):
    """Protocol for all agents in the custom grid environment.

    Any class that implements a 'get_action' method with the correct signature
    is considered an Agent.
    """

    def __init__(self, action_space: gym.spaces.Space, **kwargs):
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


class BaseAgent:
    """Base class for all agents to provide a default constructor.

    Agents can inherit from this class to satisfy the Agent protocol without
    re-implementing the constructor.
    """

    def __init__(self, action_space: gym.spaces.Space, **kwargs):
        """Initializes the agent with the given action space.

        Args:
            action_space (gym.spaces.Space): The action space of the environment.
        """
        self.action_space = action_space
        self.env = kwargs.get("env")
        self.perceived_agent_pos = None
        self.perceived_ghost_pos = None
