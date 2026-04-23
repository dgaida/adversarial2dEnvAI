"""Agent that uses Value Iteration to find the optimal path to the goal."""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import gymnasium as gym
from .base_agent import BaseAgent


class ValueIterationAgent(BaseAgent):
    """Agent that uses Value Iteration to find the optimal path to the goal.

    This agent assumes it has full knowledge of the environment and the current goal.
    """

    def __init__(self, action_space: gym.spaces.Space, **kwargs: Any):
        super().__init__(action_space, **kwargs)
        self.planner = None
        self.V: Optional[np.ndarray] = None
        self._last_goal: Optional[Tuple[int, int]] = None

    def _find_goal(self) -> Optional[Tuple[int, int]]:
        """Finds the goal position in the environment."""
        if self.env is None:
            return None
        for r in range(self.env.rows):
            for c in range(self.env.cols):
                if self.env.grid[r, c]["is_goal"]:
                    return (r, c)
        return None

    def get_value(self, state: Tuple[int, int]) -> float:
        """Returns the value of a state.

        Args:
            state (tuple): (row, col) coordinates.

        Returns:
            float: The value of the state.
        """
        goal_pos = self._find_goal()
        if goal_pos != self._last_goal and goal_pos is not None:
            from ..planner import TaskPlanner

            if self.planner is None or self.planner.env != self.env:
                self.planner = TaskPlanner(self.env)
            self.V = self.planner.value_iteration(goal_pos)
            self._last_goal = goal_pos

        if self.V is not None:
            return float(self.V[state[0], state[1]])
        return 0.0

    def get_action(self, observation: Dict[str, Any]) -> int:
        """Returns the best action based on Value Iteration.

        Args:
            observation (dict): The current observation.

        Returns:
            int: The best action.
        """
        if self.env is None:
            return 0

        # Determine current position
        if self.perceived_agent_pos is not None:
            current_pos = tuple(self.perceived_agent_pos)
        else:
            current_pos = tuple(self.env.agent_pos)

        goal_pos = self._find_goal()
        if goal_pos is None or current_pos == goal_pos:
            return 0  # Stay or default action

        if goal_pos != self._last_goal:
            from ..planner import TaskPlanner

            if self.planner is None or self.planner.env != self.env:
                self.planner = TaskPlanner(self.env)
            self.V = self.planner.value_iteration(goal_pos)
            self._last_goal = goal_pos

        action = self.planner.get_optimal_action(current_pos, self.V)

        return action
