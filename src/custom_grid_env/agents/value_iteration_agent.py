"""Agent that uses Value Iteration to find the optimal path to the goal."""

from typing import Dict, Any
import gymnasium as gym
from .base_agent import BaseAgent


class ValueIterationAgent(BaseAgent):
    """Agent that uses Value Iteration to find the optimal path to the goal.

    This agent assumes it has full knowledge of the environment and the current goal.
    """

    def __init__(self, action_space: gym.spaces.Space, **kwargs: Any):
        super().__init__(action_space, **kwargs)
        self.planner = None

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

        # Find the goal position
        goal_pos = None
        for r in range(self.env.rows):
            for c in range(self.env.cols):
                if self.env.grid[r, c]["is_goal"]:
                    goal_pos = (r, c)
                    break
            if goal_pos:
                break

        if goal_pos is None or current_pos == goal_pos:
            return 0  # Stay or default action

        # Use TaskPlanner's value iteration logic or similar
        from ..planner import TaskPlanner

        if self.planner is None or self.planner.env != self.env:
            self.planner = TaskPlanner(self.env)
        V = self.planner.value_iteration(goal_pos)
        action = self.planner.get_optimal_action(current_pos, V)

        return action
