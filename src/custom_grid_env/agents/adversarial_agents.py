"""Adversarial agents for the CustomGrid environment."""

import gymnasium as gym
from typing import Dict, Any, List, Tuple
from .base_agent import BaseAgent


class AdversarialAgent(BaseAgent):
    """Base class for minimax and expectimax agents.

    Provides shared heuristic and state-transition logic.
    """

    def __init__(self, action_space: gym.spaces.Space, depth_limit: int = 4, **kwargs: Any):
        """Initializes the adversarial agent.

        Args:
            action_space (gym.spaces.Space): The action space.
            depth_limit (int): The depth limit for the search.
            **kwargs (Any): Additional arguments, should include 'env'.
        """
        super().__init__(action_space, **kwargs)
        self.depth_limit = depth_limit

    def _get_agent_pos(self) -> List[int]:
        """Gets the current agent position from the environment."""
        if self.perceived_agent_pos is not None:
            return list(self.perceived_agent_pos)
        return list(self.env.agent_pos)

    def _get_ghost_pos(self) -> List[int]:
        """Gets the current ghost position from the environment."""
        if self.perceived_ghost_pos is not None:
            return list(self.perceived_ghost_pos)
        return list(self.env.ghost_pos)

    def _get_goal_pos(self) -> List[int]:
        """Finds the goal position in the grid."""
        for r in range(self.env.rows):
            for c in range(self.env.cols):
                if self.env.grid[r, c]["is_goal"]:
                    return [r, c]
        return [0, 2]  # Default start pos in README

    def _heuristic(
        self,
        agent_pos: List[int],
        ghost_pos: List[int],
        terminated: bool,
        info: Dict[str, Any],
    ) -> float:
        """Heuristic function to evaluate a state.

        Args:
            agent_pos (list): Agent [row, col].
            ghost_pos (list): Ghost [row, col].
            terminated (bool): Whether the state is terminal.
            info (dict): Additional information about the state.

        Returns:
            float: Heuristic value.
        """
        if terminated:
            if info.get("reached_goal"):
                return 10000.0
            if info.get("caught_by_ghost"):
                return -10000.0

        goal_pos = self._get_goal_pos()
        dist_to_goal = self.env._calculate_shortest_path_distance(agent_pos, goal_pos)
        dist_to_ghost = self.env._calculate_shortest_path_distance(agent_pos, ghost_pos)

        if dist_to_goal == -1:
            dist_to_goal = self.env.rows * self.env.cols
        if dist_to_ghost == -1:
            dist_to_ghost = self.env.rows * self.env.cols

        # Maximize distance to ghost and minimize distance to goal
        score = -dist_to_goal * 10

        if dist_to_ghost <= 2:
            score -= (4 - dist_to_ghost) * 100
        else:
            score += dist_to_ghost * 2

        return float(score)

    def _get_next_state(
        self, agent_pos: List[int], ghost_pos: List[int], turn: int, action: int
    ) -> Tuple[List[int], List[int], bool, Dict[str, Any]]:
        """Simulates a state transition.

        Args:
            agent_pos (list): Agent [row, col].
            ghost_pos (list): Ghost [row, col].
            turn (int): 0 for agent, 1 for ghost.
            action (int): Action to take.

        Returns:
            tuple: (new_agent_pos, new_ghost_pos, terminated, info)
        """
        new_agent_pos = list(agent_pos)
        new_ghost_pos = list(ghost_pos)

        if turn == 0:
            new_agent_pos = self.env._move_entity(agent_pos, action)
        else:
            new_ghost_pos = self.env._move_entity(ghost_pos, action)

        terminated = False
        info = {}
        if new_agent_pos == new_ghost_pos:
            terminated = True
            info["caught_by_ghost"] = True
        elif self.env.grid[new_agent_pos[0], new_agent_pos[1]]["is_goal"]:
            terminated = True
            info["reached_goal"] = True

        return new_agent_pos, new_ghost_pos, terminated, info

    def _get_agent_outcomes(
        self, agent_pos: List[int], action: int
    ) -> List[Tuple[List[int], float]]:
        """Returns possible agent positions after action and their probabilities."""
        prob = self.env.slip_probability
        slip_type = self.env.slip_type

        outcomes = []
        if slip_type == "longitudinal":
            # Stay: prob/2
            outcomes.append((self.env._move_entity(agent_pos, -1), prob / 2.0))
            # Move 1: 1 - prob
            outcomes.append((self.env._move_entity(agent_pos, action), 1.0 - prob))
            # Move 2: prob/2
            pos2 = self.env._move_entity(agent_pos, action)
            pos2 = self.env._move_entity(pos2, action)
            outcomes.append((pos2, prob / 2.0))
        elif slip_type == "perpendicular":
            perpendicular = {0: [1, 3], 1: [0, 2], 2: [1, 3], 3: [0, 2]}
            # Intended: 1 - prob
            outcomes.append((self.env._move_entity(agent_pos, action), 1.0 - prob))
            # Sides: prob/2 each
            for side in perpendicular[action]:
                outcomes.append((self.env._move_entity(agent_pos, side), prob / 2.0))
        else:
            outcomes.append((self.env._move_entity(agent_pos, action), 1.0))

        return outcomes


class MinimaxAgent(AdversarialAgent):
    """Agent that uses Minimax with Alpha-Beta pruning."""

    def get_action(self, observation: Dict[str, Any]) -> int:
        """Returns the best action using minimax.

        Args:
            observation (dict): The current observation.

        Returns:
            int: The best action.
        """
        agent_pos = self._get_agent_pos()
        ghost_pos = self._get_ghost_pos()

        # Check if we are acting as the ghost
        is_ghost = "agent_relative_pos" in observation

        alpha = -float("inf")
        beta = float("inf")

        actions = [0, 1, 2, 3]

        if not is_ghost:
            best_score = -float("inf")
            best_action = 0
            for action in actions:
                new_ap, new_gp, term, info = self._get_next_state(
                    agent_pos, ghost_pos, 0, action
                )
                score = self._min_value(new_ap, new_gp, 1, term, info, alpha, beta)
                if score > best_score:
                    best_score = score
                    best_action = action
                alpha = max(alpha, best_score)
        else:
            best_score = float("inf")
            best_action = 0
            for action in actions:
                new_ap, new_gp, term, info = self._get_next_state(
                    agent_pos, ghost_pos, 1, action
                )
                score = self._max_value(new_ap, new_gp, 1, term, info, alpha, beta)
                if score < best_score:
                    best_score = score
                    best_action = action
                beta = min(beta, best_score)

        return best_action

    def _max_value(self, agent_pos, ghost_pos, depth, terminated, info, alpha, beta):
        """Calculates the maximum value for the minimax algorithm."""
        if terminated or depth >= self.depth_limit:
            return self._heuristic(agent_pos, ghost_pos, terminated, info)

        v = -float("inf")
        for action in [0, 1, 2, 3]:
            new_ap, new_gp, term, info_next = self._get_next_state(
                agent_pos, ghost_pos, 0, action
            )
            v = max(
                v,
                self._min_value(
                    new_ap, new_gp, depth + 1, term, info_next, alpha, beta
                ),
            )
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def _min_value(self, agent_pos, ghost_pos, depth, terminated, info, alpha, beta):
        """Calculates the minimum value for the minimax algorithm."""
        if terminated or depth >= self.depth_limit:
            return self._heuristic(agent_pos, ghost_pos, terminated, info)

        v = float("inf")
        for action in [0, 1, 2, 3]:
            new_ap, new_gp, term, info_next = self._get_next_state(
                agent_pos, ghost_pos, 1, action
            )
            v = min(
                v,
                self._max_value(
                    new_ap, new_gp, depth + 1, term, info_next, alpha, beta
                ),
            )
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v


class ExpectimaxAgent(AdversarialAgent):
    """Agent that uses Expectimax algorithm."""

    def get_action(self, observation: Dict[str, Any]) -> int:
        """Returns the best action using expectimax.

        Args:
            observation (dict): The current observation.

        Returns:
            int: The best action.
        """
        agent_pos = self._get_agent_pos()
        ghost_pos = self._get_ghost_pos()

        is_ghost = "agent_relative_pos" in observation

        actions = [0, 1, 2, 3]

        if not is_ghost:
            best_score = -float("inf")
            best_action = 0
            for action in actions:
                outcomes = self._get_agent_outcomes(agent_pos, action)
                score = 0
                for next_ap, prob in outcomes:
                    term = False
                    info = {}
                    if next_ap == ghost_pos:
                        term = True
                        info["caught_by_ghost"] = True
                    elif self.env.grid[next_ap[0], next_ap[1]]["is_goal"]:
                        term = True
                        info["reached_goal"] = True

                    score += prob * self._expect_value(
                        next_ap, ghost_pos, 1, term, info
                    )

                if score > best_score:
                    best_score = score
                    best_action = action
        else:
            best_score = float("inf")
            best_action = 0
            for action in actions:
                new_ap, new_gp, term, info = self._get_next_state(
                    agent_pos, ghost_pos, 1, action
                )
                score = self._max_value(new_ap, new_gp, 1, term, info)
                if score < best_score:
                    best_score = score
                    best_action = action
        return best_action

    def _max_value(self, agent_pos, ghost_pos, depth, terminated, info):
        """Calculates the maximum value for the expectimax algorithm."""
        if terminated or depth >= self.depth_limit:
            return self._heuristic(agent_pos, ghost_pos, terminated, info)

        v = -float("inf")
        for action in [0, 1, 2, 3]:
            outcomes = self._get_agent_outcomes(agent_pos, action)
            score = 0
            for next_ap, prob in outcomes:
                term = False
                info_next = {}
                if next_ap == ghost_pos:
                    term = True
                    info_next["caught_by_ghost"] = True
                elif self.env.grid[next_ap[0], next_ap[1]]["is_goal"]:
                    term = True
                    info_next["reached_goal"] = True
                score += prob * self._expect_value(
                    next_ap, ghost_pos, depth + 1, term, info_next
                )
            v = max(v, score)
        return v

    def _expect_value(self, agent_pos, ghost_pos, depth, terminated, info):
        """Calculates the expected value for the expectimax algorithm."""
        if terminated or depth >= self.depth_limit:
            return self._heuristic(agent_pos, ghost_pos, terminated, info)

        v = 0
        actions = [0, 1, 2, 3]
        for action in actions:
            new_ap, new_gp, term, info_next = self._get_next_state(
                agent_pos, ghost_pos, 1, action
            )
            v += self._max_value(new_ap, new_gp, depth + 1, term, info_next)
        return v / len(actions)
