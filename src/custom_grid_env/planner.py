"""Handles task interpretation using LLM and optimal path planning."""

import json
import re
import numpy as np
from typing import List, Tuple, Dict, Any
from llm_client import LLMClient
from .env import CustomGridEnv
from .logger import get_logger

logger = get_logger(__name__)


class TaskPlanner:
    """Handles task interpretation using LLM and optimal path planning."""

    def __init__(self, env: CustomGridEnv, **kwargs):
        """Initializes the TaskPlanner.

        Args:
            env (CustomGridEnv): The environment instance.
            **kwargs: Additional arguments for LLMClient.
        """
        self.env = env
        # Default settings if not provided in kwargs
        llm = kwargs.get("llm", "qwen/qwen3-32b")
        api_choice = kwargs.get("api_choice", "groq")

        # In tests, we might not have API keys, so we might want to skip LLM initialization
        # or use a mock. But for now, we try to initialize it.
        try:
            self.llm_client = LLMClient(llm=llm, api_choice=api_choice, **kwargs)
        except Exception as e:
            logger.warning(
                f"Could not initialize LLMClient: {e}. Some features may not work."
            )
            self.llm_client = None

    def identify_targets(
        self, task_description: str
    ) -> Tuple[List[Tuple[int, int]], str]:
        """Identifies target coordinates from a natural language task description.

        Args:
            task_description (str): The task description.

        Returns:
            Tuple[List[Tuple[int, int]], str]: List of (row, col) coordinates and raw response.
        """
        if self.llm_client is None:
            logger.error("LLMClient not initialized. Cannot identify targets.")
            return [], "LLMClient not initialized."

        grid_desc = self.env.get_grid_description()

        system_prompt = (
            "Du bist ein hilfreicher Assistent, der Navigationsaufgaben in einem 4x5 Grid interpretiert. Deine einzige Aufgabe ist es, die Koordinaten der im Text genannten Felder zu identifizieren. Ein nachgelagerter TSP-Solver wird die optimale Reihenfolge berechnen, daher musst du die Felder lediglich in der Reihenfolge ihrer Nennung auflisten. Plane KEINE optimale Tour und ändere NICHT die Reihenfolge der Felder. Gib die Koordinaten genau in der Reihenfolge zurück, in der sie in der Aufgabenstellung erscheinen. Versuche NICHT, den Pfad zu optimieren oder den Ausgangsort einzubeziehen, sofern er nicht explizit als zu besuchendes Feld genannt wurde. "
            "Hier ist die Beschreibung des Grids:\n"
            f"{grid_desc}\n"
            "Gib die Koordinaten der zu besuchenden Felder in der Reihenfolge zurück, "
            "in der sie im Text genannt werden. "
            "Antworte AUSSCHLIESSLICH mit einem JSON-Array von Listen, z.B. [[0, 1], [2, 3]]. Erkläre nichts, schreibe keinen Text vor oder nach dem JSON und nutze absolut keine <think> Tags."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task_description},
        ]

        logger.debug(f"System prompt: {system_prompt}")
        logger.debug(f"Task description: {task_description}")

        try:
            response = self.llm_client.chat_completion(messages)
            logger.debug(f"Raw LLM response: {response}")

            # Remove <think>...</think> blocks
            clean_response = re.sub(
                r"<think>.*?</think>", "", response, flags=re.DOTALL
            )
            # Remove anything before the last </think> tag if it still exists (e.g. truncated response)
            if "</think>" in clean_response:
                clean_response = clean_response.split("</think>")[-1]
            # Remove any unclosed <think> tag at the beginning
            if "<think>" in clean_response:
                clean_response = clean_response.split("<think>")[-1]

            # Extract JSON array using regex for robustness
            # Look for [[x, y], [a, b]] pattern
            match = re.search(r"\[\s*\[.*\]\s*\]", clean_response, re.DOTALL)
            if match:
                clean_response = match.group(0)
            else:
                # Fallback to markdown blocks if regex for array fails
                clean_response = clean_response.strip()
                if "```json" in clean_response:
                    json_match = re.search(
                        r"```json\s*(.*?)\s*```", clean_response, re.DOTALL
                    )
                    if json_match:
                        clean_response = json_match.group(1)
                elif "```" in clean_response:
                    code_match = re.search(
                        r"```\s*(.*?)\s*```", clean_response, re.DOTALL
                    )
                    if code_match:
                        clean_response = code_match.group(1)

            # If after all cleaning it still doesn't look like a JSON array,
            # don't even try to parse it to avoid noisy error logs
            if not clean_response.strip().startswith("["):
                logger.warning(
                    f"No JSON array found in LLM response: {clean_response[:100]}..."
                )
                return [], response

            targets = json.loads(clean_response)
            return [tuple(t) for t in targets], response
        except Exception as e:
            logger.error(f"Error identifying targets: {e}")
            logger.error(
                f"Response that failed: {response if 'response' in locals() else 'N/A'}"
            )
            if "clean_response" in locals():
                logger.error(f"Cleaned response that failed: {clean_response}")
            return [], response if "response" in locals() else str(e)

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
