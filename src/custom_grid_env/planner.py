import numpy as np
from typing import List, Tuple
import json
from llm_client import LLMClient
from .env import CustomGridEnv
from .logger import get_logger

logger = get_logger(__name__)


class TaskPlanner:
    """Handles task interpretation using LLM and optimal path planning."""

    def __init__(self, env: CustomGridEnv):
        """Initializes the TaskPlanner.

        Args:
            env (CustomGridEnv): The environment instance.
        """
        self.env = env
        self.llm_client = LLMClient()

    def identify_targets(self, task_description: str) -> List[Tuple[int, int]]:
        """Identifies target coordinates from a natural language task description.

        Args:
            task_description (str): The task description.

        Returns:
            List[Tuple[int, int]]: List of (row, col) coordinates to visit.
        """
        grid_desc = self.env.get_grid_description()

        system_prompt = (
            "Du bist ein hilfreicher Assistent, der Navigationsaufgaben in einem 4x5 Grid interpretiert. "
            "Hier ist die Beschreibung des Grids:\n"
            f"{grid_desc}\n"
            "Gib die Koordinaten der zu besuchenden Felder in der Reihenfolge zurück, "
            "in der sie im Text genannt werden. "
            "Gib NUR ein JSON-Array von Listen zurück, z.B. [[0, 1], [2, 3]]."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task_description},
        ]

        try:
            response = self.llm_client.chat_completion(messages)
            # Basic cleaning if LLM adds markdown
            clean_response = response.strip()
            if clean_response.startswith("```json"):
                clean_response = clean_response[7:-3].strip()
            elif clean_response.startswith("```"):
                clean_response = clean_response[3:-3].strip()

            targets = json.loads(clean_response)
            return [tuple(t) for t in targets]
        except Exception as e:
            logger.error(f"Error identifying targets: {e}")
            return []

    def value_iteration(
        self, goal_pos: Tuple[int, int], theta: float = 0.0001
    ) -> np.ndarray:
        """Computes optimal values for each cell using value iteration.

        Args:
            goal_pos (Tuple[int, int]): The target position.
            theta (float): Convergence threshold.

        Returns:
            np.ndarray: Value function.
        """
        V = np.zeros((self.env.rows, self.env.cols))
        gamma = 0.9  # Discount factor

        while True:
            delta = 0
            for r in range(self.env.rows):
                for c in range(self.env.cols):
                    v = V[r, c]
                    if (r, c) == goal_pos:
                        V[r, c] = 100.0
                        delta = max(delta, abs(v - V[r, c]))
                        continue

                    action_values = []
                    for action in range(4):
                        next_pos = self.env._move_entity([r, c], action)
                        reward = -1.0  # Step penalty
                        action_values.append(
                            reward + gamma * V[next_pos[0], next_pos[1]]
                        )

                    V[r, c] = max(action_values)
                    delta = max(delta, abs(v - V[r, c]))

            if delta < theta:
                break
        return V

    def get_optimal_action(self, current_pos: Tuple[int, int], V: np.ndarray) -> int:
        """Determines the best action from current_pos based on value function V.

        Args:
            current_pos (Tuple[int, int]): Current position.
            V (np.ndarray): Value function.

        Returns:
            int: Best action (0-3).
        """
        action_values = []
        for action in range(4):
            next_pos = self.env._move_entity(list(current_pos), action)
            # If we are already at the goal, any action might be returned.
            # But the goal itself should have the highest value.
            action_values.append(V[next_pos[0], next_pos[1]])

        # Prefer actions that actually change position if possible,
        # but here we just take the max value.
        best_action = int(np.argmax(action_values))
        return best_action

    def solve_tsp(
        self, start_pos: Tuple[int, int], targets: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """Solves a simple TSP to find the optimal visiting order.

        Args:
            start_pos (Tuple[int, int]): Starting position.
            targets (List[Tuple[int, int]]): Target positions.

        Returns:
            List[Tuple[int, int]]: Optimal order of targets.
        """
        import itertools

        if not targets:
            return []

        all_points = [start_pos] + targets
        # Precompute distances using BFS (shortest path)
        dist_matrix = {}
        for p1 in all_points:
            for p2 in all_points:
                dist_matrix[(p1, p2)] = self.env._calculate_shortest_path_distance(
                    list(p1), list(p2)
                )

        best_perm = None
        min_dist = float("inf")

        for perm in itertools.permutations(targets):
            current_dist = 0
            curr = start_pos
            for target in perm:
                current_dist += dist_matrix[(curr, target)]
                curr = target
            current_dist += dist_matrix[(curr, start_pos)]  # Return to start

            if current_dist < min_dist:
                min_dist = current_dist
                best_perm = perm

        return list(best_perm) if best_perm else []

    def get_path(
        self, start_pos: Tuple[int, int], goal_pos: Tuple[int, int]
    ) -> List[int]:
        """Gets the list of actions for the optimal path from start to goal.

        Args:
            start_pos (Tuple[int, int]): Start position.
            goal_pos (Tuple[int, int]): Goal position.

        Returns:
            List[int]: List of actions.
        """
        if start_pos == goal_pos:
            return []

        V = self.value_iteration(goal_pos)
        path = []
        curr = start_pos
        max_steps = self.env.rows * self.env.cols * 2  # Safety margin
        steps = 0
        while tuple(curr) != goal_pos and steps < max_steps:
            action = self.get_optimal_action(tuple(curr), V)
            new_pos = self.env._move_entity(list(curr), action)
            if tuple(new_pos) == tuple(curr):
                # We are stuck? This should not happen with value iteration on this grid
                # unless there is no path.
                break
            path.append(action)
            curr = new_pos
            steps += 1
        return path
