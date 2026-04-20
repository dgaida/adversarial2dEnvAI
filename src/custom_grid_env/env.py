"""Gymnasium environment for the CustomGrid task."""

import gymnasium as gym
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from .renderer import PygameRenderer
from .logger import get_logger

logger = get_logger(__name__)


class CustomGridEnv(gym.Env):
    """A custom grid environment with an agent and a ghost.

    Metadata:
        render_modes (list): Supported render modes.
        render_fps (int): Rendering frames per second.

    Attributes:
        render_mode (str): Current render mode.
        slip_probability (float): Probability of slipping to a perpendicular direction.
        rows (int): Number of rows in the grid.
        cols (int): Number of columns in the grid.
        grid (np.ndarray): The grid containing cell information.
        agent_pos (list): Current position of the agent [row, col].
        start_pos (list): Starting position of the agent.
        ghost_pos (list): Current position of the ghost.
        ghost_start_pos (list): Starting position of the ghost.
        step_count (int): Current step count in the episode.
        current_turn (int): Whose turn it is (0 for agent, 1 for ghost).
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        render_mode: str = "human",
        slip_probability: float = 0.2,
        slip_type: str = "longitudinal",
        color_sensor_quality: float = 0.8,
        deterministic: bool = False,
        use_ghost: bool = True,
    ):
        """Initializes the CustomGridEnv.

        Args:
            render_mode (str): The mode to render with. Defaults to "human".
            slip_probability (float): Chance to move perpendicular to intended direction. Defaults to 0.2.
            slip_type (str): Type of slipping ("perpendicular" or "longitudinal").
                Defaults to "longitudinal".
            color_sensor_quality (float): Probability of the color sensor measuring the correct color.
                Defaults to 0.8.
        """
        super().__init__()
        self.render_mode = render_mode
        self.slip_probability = slip_probability
        self.slip_type = slip_type
        self.color_sensor_quality = color_sensor_quality
        self.deterministic = deterministic
        self.use_ghost = use_ghost
        self.rows = 4
        self.cols = 5
        self.observation_space = gym.spaces.Dict(
            {
                "current_cell": gym.spaces.Dict(
                    {
                        "colour": gym.spaces.Discrete(3),  # 0=none, 1=red, 2=green
                        "has_item": gym.spaces.MultiBinary(3),  # [dog, flower, notes]
                        "is_goal": gym.spaces.Discrete(2),
                        "text": gym.spaces.Text(max_length=10),
                    }
                ),
                "neighbors": gym.spaces.Dict(
                    {
                        "up": gym.spaces.Dict(
                            {
                                "accessible": gym.spaces.Discrete(2),
                                "colour": gym.spaces.Discrete(3),
                            }
                        ),
                        "right": gym.spaces.Dict(
                            {
                                "accessible": gym.spaces.Discrete(2),
                                "colour": gym.spaces.Discrete(3),
                            }
                        ),
                        "down": gym.spaces.Dict(
                            {
                                "accessible": gym.spaces.Discrete(2),
                                "colour": gym.spaces.Discrete(3),
                            }
                        ),
                        "left": gym.spaces.Dict(
                            {
                                "accessible": gym.spaces.Discrete(2),
                                "colour": gym.spaces.Discrete(3),
                            }
                        ),
                    }
                ),
                "ghost_relative_pos": gym.spaces.Box(
                    low=-4, high=4, shape=(2,), dtype=np.int32
                ),
                "ghost_distance": gym.spaces.Discrete(21),  # Max distance in 4x5 grid
            }
        )

        self.action_space = gym.spaces.Discrete(4)  # 0: left, 1: down, 2: right, 3: up

        self.grid = np.empty((self.rows, self.cols), dtype=object)
        self._setup_grid()
        self._setup_walls()
        self.agent_pos = [0, 2]
        self.start_pos = [0, 2]
        self.ghost_pos = [3, 4]
        self.ghost_start_pos = [3, 4]
        self.step_count = 0
        self.current_turn = 0
        self.info = {}

        # Rendering setup
        self.renderer = None
        if self.render_mode in self.metadata["render_modes"]:
            self.renderer = PygameRenderer(
                rows=self.rows,
                cols=self.cols,
                render_mode=self.render_mode,
                render_fps=self.metadata["render_fps"],
            )

    def _setup_grid(self):
        """Sets up the initial grid contents."""
        self.grid[0, 0] = {
            "colour": 2,
            "items": ["dog"],
            "is_goal": False,
            "is_start": False,
            "text": "",
        }
        self.grid[0, 1] = {
            "colour": 1,
            "items": [],
            "is_goal": False,
            "is_start": False,
            "text": "",
        }
        self.grid[0, 2] = {
            "colour": 0,
            "items": [],
            "is_goal": False,
            "is_start": True,
            "text": "Start",
        }
        self.grid[0, 3] = {
            "colour": 0,
            "items": [],
            "is_goal": False,
            "is_start": False,
            "text": "",
        }
        self.grid[0, 4] = {
            "colour": 0,
            "items": ["dog", "two_notes"],
            "is_goal": False,
            "is_start": False,
            "text": "",
        }

        self.grid[1, 0] = {
            "colour": 0,
            "items": [],
            "is_goal": False,
            "is_start": False,
            "text": "",
        }
        self.grid[1, 1] = {
            "colour": 2,
            "items": [],
            "is_goal": False,
            "is_start": False,
            "text": "",
        }
        self.grid[1, 2] = {
            "colour": 0,
            "items": [],
            "is_goal": False,
            "is_start": False,
            "text": "",
        }
        self.grid[1, 3] = {
            "colour": 0,
            "items": ["flower"],
            "is_goal": False,
            "is_start": False,
            "text": "",
        }
        self.grid[1, 4] = {
            "colour": 2,
            "items": [],
            "is_goal": False,
            "is_start": False,
            "text": "",
        }

        self.grid[2, 0] = {
            "colour": 1,
            "items": [],
            "is_goal": False,
            "is_start": False,
            "text": "",
        }
        self.grid[2, 1] = {
            "colour": 2,
            "items": [],
            "is_goal": False,
            "is_start": False,
            "text": "",
        }
        self.grid[2, 2] = {
            "colour": 0,
            "items": ["one_note"],
            "is_goal": False,
            "is_start": False,
            "text": "",
        }
        self.grid[2, 3] = {
            "colour": 1,
            "items": [],
            "is_goal": False,
            "is_start": False,
            "text": "",
        }
        self.grid[2, 4] = {
            "colour": 0,
            "items": [],
            "is_goal": False,
            "is_start": False,
            "text": "",
        }

        self.grid[3, 0] = {
            "colour": 1,
            "items": [],
            "is_goal": False,
            "is_start": False,
            "text": "",
        }
        self.grid[3, 1] = {
            "colour": 0,
            "items": [],
            "is_goal": True,
            "is_start": False,
            "text": "Ziel",
        }
        self.grid[3, 2] = {
            "colour": 2,
            "items": ["two_notes"],
            "is_goal": False,
            "is_start": False,
            "text": "",
        }

        self.grid[3, 3] = {
            "colour": 0,
            "items": ["two_flowers"],
            "is_goal": False,
            "is_start": False,
            "text": "",
        }
        self.grid[3, 4] = {
            "colour": 0,
            "items": [],
            "is_goal": True,
            "is_start": False,
            "text": "Stopp",
        }

    def _setup_walls(self):
        """Sets up the walls in the environment."""
        self.walls_horizontal = np.zeros((self.rows, self.cols), dtype=bool)
        self.walls_vertical = np.zeros((self.rows, self.cols), dtype=bool)

        self.walls_horizontal[0, 3] = True
        self.walls_horizontal[1, 2] = True
        self.walls_horizontal[1, 3] = True
        self.walls_horizontal[2, 3] = True

        self.walls_vertical[0, 0] = True
        self.walls_vertical[1, 2] = True
        self.walls_vertical[2, 1] = True
        self.walls_vertical[2, 2] = True
        self.walls_vertical[3, 1] = True

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Resets the environment.

        Args:
            seed (int, optional): Random seed.
            options (dict, optional): Additional options.

        Returns:
            tuple: (observation, info)
        """
        super().reset(seed=seed)
        self.agent_pos = list(self.start_pos)
        self.ghost_pos = list(self.ghost_start_pos)
        self.step_count = 0
        self.current_turn = 0
        self.info = {
            "current_turn": "agent",
            "color_measurement": self._get_color_sensor_measurement(self.agent_pos),
            "ghost_distance": self._calculate_shortest_path_distance(
                self.agent_pos, self.ghost_pos
            ),
        }
        return self._get_obs(), self.info

    def _move_entity(self, current_pos: List[int], action: int) -> List[int]:
        """Helper function to move an entity with wall checking.

        Args:
            current_pos (list): Current [row, col] position.
            action (int): Action to take.

        Returns:
            list: New [row, col] position.
        """
        row, col = current_pos
        new_row, new_col = row, col

        if action == 0:
            new_col = col - 1
        elif action == 1:
            new_row = row + 1
        elif action == 2:
            new_col = col + 1
        elif action == 3:
            new_row = row - 1

        move_valid = True

        if new_row < 0 or new_row >= self.rows or new_col < 0 or new_col >= self.cols:
            move_valid = False

        if move_valid:
            move_valid = self._is_move_valid(current_pos, [new_row, new_col])

        if move_valid:
            return [new_row, new_col]
        else:
            return current_pos

    def _get_ghost_obs(self) -> Dict[str, Any]:
        """Returns observation from the ghost's perspective.

        Returns:
            dict: Ghost's observation.
        """
        row, col = self.ghost_pos
        current_cell = self.grid[row, col]

        current_obs = {
            "colour": current_cell["colour"],
            "has_item": np.array(
                [
                    1 if any("dog" in i for i in current_cell["items"]) else 0,
                    1 if any("flower" in i for i in current_cell["items"]) else 0,
                    (
                        1
                        if any(
                            note in current_cell["items"]
                            for note in ["one_note", "two_notes"]
                        )
                        else 0
                    ),
                ],
                dtype=np.int8,
            ),
            "is_goal": 1 if current_cell["is_goal"] else 0,
        }

        neighbors_obs = {}
        for direction, (dr, dc) in [
            ("up", (-1, 0)),
            ("right", (0, 1)),
            ("down", (1, 0)),
            ("left", (0, -1)),
        ]:
            neighbor_pos = [row + dr, col + dc]
            accessible = self._is_move_valid(self.ghost_pos, neighbor_pos)
            neighbor_cell = self._get_cell_info(row + dr, col + dc)
            neighbors_obs[direction] = {
                "accessible": 1 if accessible else 0,
                "colour": neighbor_cell["colour"] if neighbor_cell else 0,
            }

        agent_relative = np.array(
            [
                self.agent_pos[0] - self.ghost_pos[0],
                self.agent_pos[1] - self.ghost_pos[1],
            ],
            dtype=np.int32,
        )

        return {
            "current_cell": current_obs,
            "neighbors": neighbors_obs,
            "agent_relative_pos": agent_relative,
            "agent_distance": self._calculate_shortest_path_distance(
                self.ghost_pos, self.agent_pos
            ),
        }

    def move_ghost(self, ghost_action: int):
        """Moves the ghost with an externally provided action.

        Args:
            ghost_action (int): Action for ghost (0=left, 1=down, 2=right, 3=up).
        """
        self.ghost_pos = self._move_entity(self.ghost_pos, ghost_action)

    def get_reward_structure(self) -> Dict[str, Any]:
        """Gets the reward structure for this environment.

        Returns:
            dict: Dictionary describing all rewards and their values.
        """
        return {
            "step_penalty": -1,
            "caught_by_ghost": -50,
            "reached_goal": 100,
            "slip_probability": self.slip_probability,
            "terminal_states": ["caught_by_ghost", "reached_goal"],
        }

    def calculate_reward(
        self, caught_by_ghost: bool = False
    ) -> Tuple[float, bool, Dict[str, Any]]:
        """Calculates reward based on current game state.

        Args:
            caught_by_ghost (bool): Whether the ghost caught the agent. Defaults to False.

        Returns:
            tuple: (reward, terminated, info_dict)
        """
        reward_structure = self.get_reward_structure()
        reward = reward_structure["step_penalty"]
        terminated = False

        if caught_by_ghost:
            return (
                float(reward_structure["caught_by_ghost"]),
                True,
                {"caught_by_ghost": True},
            )

        current_cell = self.grid[self.agent_pos[0], self.agent_pos[1]]

        if current_cell["is_goal"]:
            return float(reward_structure["reached_goal"]), True, {"reached_goal": True}

        return float(reward), terminated, {}

    def _get_color_sensor_measurement(self, pos: List[int]) -> int:
        """Measures the color of the ground at the given position with noise.

        The sensor measures the correct color with `color_sensor_quality` probability and a wrong
        color with `(1 - color_sensor_quality) / 2` probability for each of the other two colors.

        Args:
            pos (list): [row, col] position to measure.

        Returns:
            int: Measured color (0=white, 1=red, 2=green).
        """
        actual_color = self.grid[pos[0], pos[1]]["colour"]
        prob = self.np_random.random()

        error_prob = (1.0 - self.color_sensor_quality) / 2.0

        if prob < self.color_sensor_quality:
            return actual_color
        elif prob < self.color_sensor_quality + error_prob:
            # First wrong color
            return (actual_color + 1) % 3
        else:
            # Second wrong color
            return (actual_color + 2) % 3

    def _apply_slip(self, intended_action: int) -> Tuple[List[int], bool]:
        """Applies slip probability to potentially change the action or multiple actions.

        Args:
            intended_action (int): The action the agent intended to take.

        Returns:
            tuple: (list of actual_actions, slipped)
        """
        if self.deterministic or self.slip_probability <= 0:
            return [intended_action], False

        if self.slip_type == "longitudinal":
            prob = self.np_random.random()
            if prob < self.slip_probability / 2:
                # Stay in place (0 steps)
                return [], True
            elif prob < self.slip_probability:
                # Move twice (2 steps)
                return [intended_action, intended_action], True
            else:
                return [intended_action], False
        else:
            # Default to perpendicular
            perpendicular = {
                0: [1, 3],
                1: [0, 2],
                2: [1, 3],
                3: [0, 2],
            }

            if self.np_random.random() < self.slip_probability:
                actual_action = int(
                    self.np_random.choice(perpendicular[intended_action])
                )
                return [actual_action], True

            return [intended_action], False

    def step(
        self, action: int
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Executes one step in the environment.

        Args:
            action (int): Action for current entity.

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        logger.debug(f"step(action={action}) called. current_turn={self.current_turn}")

        # Preserve certain keys across info.clear() if needed
        # Actually, if the user calls render() manually, cnn_prediction might be in self.info.
        # But step() usually clears it for the new step.
        info = self.info
        logger.debug(f"info before clear: {info}")

        preserved_info = {}
        keys_to_preserve = [
            "cnn_prediction",
            "cnn_probs",
            "estimated_pos",
            "color_measurement",
            "intended_action",
            "actual_action",
            "slipped",
            "particles",
            "agent_values",
            "ghost_values",
            "reached_goal",
            "caught_by_ghost",
        ]
        for key in keys_to_preserve:
            if key in info:
                preserved_info[key] = info[key]

        info.clear()
        info.update(preserved_info)

        reward = 0.0
        terminated = False
        action_names = {0: "left", 1: "down", 2: "right", 3: "up"}

        if self.current_turn == 0:
            self.step_count += 1
            actual_actions, slipped = self._apply_slip(action)
            for act in actual_actions:
                self.agent_pos = self._move_entity(self.agent_pos, act)
                logger.debug(f"Agent moved to {self.agent_pos} via action {act}")

            info["slipped"] = slipped
            info["intended_action"] = action_names[action]
            info["actual_action"] = (
                ", ".join([action_names[a] for a in actual_actions])
                if actual_actions
                else "stay"
            )
            info["color_measurement"] = self._get_color_sensor_measurement(
                self.agent_pos
            )

            current_cell = self.grid[self.agent_pos[0], self.agent_pos[1]]
            if self.use_ghost and self.agent_pos == self.ghost_pos:
                reward = float(self.get_reward_structure()["caught_by_ghost"])
                terminated = True
                info["caught_by_ghost"] = True
            elif current_cell["is_goal"]:
                reward = float(self.get_reward_structure()["reached_goal"])
                terminated = True
                info["reached_goal"] = True
            else:
                reward = float(self.get_reward_structure()["step_penalty"])

            if self.use_ghost:
                self.current_turn = 1
                info["current_turn"] = "ghost"
            else:
                self.current_turn = 0
                info["current_turn"] = "agent"
            info["mover"] = "agent"
            info["ghost_distance"] = self._calculate_shortest_path_distance(
                self.agent_pos, self.ghost_pos
            )
            self.info = info

        else:
            self.move_ghost(action)
            logger.debug(f"Ghost moved to {self.ghost_pos} via action {action}")
            if self.use_ghost and self.agent_pos == self.ghost_pos:
                reward = float(self.get_reward_structure()["caught_by_ghost"])
                terminated = True
                info["caught_by_ghost"] = True

            self.current_turn = 0
            info["current_turn"] = "agent"
            info["mover"] = "ghost"
            info["ghost_distance"] = self._calculate_shortest_path_distance(
                self.agent_pos, self.ghost_pos
            )

        self.info = info
        return self._get_obs(), float(reward), terminated, False, info

    def get_current_turn(self) -> str:
        """Returns whose turn it is.

        Returns:
            str: 'agent' or 'ghost'.
        """
        return "agent" if self.current_turn == 0 else "ghost"

    def _get_cell_info(self, row: int, col: int) -> Optional[Dict[str, Any]]:
        """Gets information about a specific cell.

        Args:
            row (int): Cell row.
            col (int): Cell column.

        Returns:
            dict, optional: Cell information or None if out of bounds.
        """
        if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
            return None
        return self.grid[row, col]

    def _is_move_valid(self, from_pos: List[int], to_pos: List[int]) -> bool:
        """Checks if a move is valid.

        Args:
            from_pos (list): Starting [row, col].
            to_pos (list): Target [row, col].

        Returns:
            bool: True if move is valid, False otherwise.
        """
        from_row, from_col = from_pos
        to_row, to_col = to_pos

        if to_row < 0 or to_row >= self.rows or to_col < 0 or to_col >= self.cols:
            return False

        if to_row < from_row:
            return not self.walls_horizontal[from_row - 1, from_col]
        elif to_row > from_row:
            return not self.walls_horizontal[from_row, from_col]
        elif to_col < from_col:
            return not self.walls_vertical[from_row, from_col - 1]
        elif to_col > from_col:
            return not self.walls_vertical[from_row, from_col]

        return True

    def _get_obs(self) -> Dict[str, Any]:
        """Returns detailed observation about current cell and neighbors.

        Returns:
            dict: Detailed observation.
        """
        row, col = self.agent_pos
        current_cell = self.grid[row, col]

        current_obs = {
            "colour": current_cell["colour"],
            "has_item": np.array(
                [
                    1 if any("dog" in i for i in current_cell["items"]) else 0,
                    1 if any("flower" in i for i in current_cell["items"]) else 0,
                    (
                        1
                        if any(
                            note in current_cell["items"]
                            for note in ["one_note", "two_notes"]
                        )
                        else 0
                    ),
                ],
                dtype=np.int8,
            ),
            "is_goal": 1 if current_cell["is_goal"] else 0,
            "text": current_cell["text"],
        }

        neighbors_obs = {}
        for direction, (dr, dc) in [
            ("up", (-1, 0)),
            ("right", (0, 1)),
            ("down", (1, 0)),
            ("left", (0, -1)),
        ]:
            neighbor_pos = [row + dr, col + dc]
            accessible = self._is_move_valid(self.agent_pos, neighbor_pos)
            neighbor_cell = self._get_cell_info(row + dr, col + dc)
            neighbors_obs[direction] = {
                "accessible": 1 if accessible else 0,
                "colour": neighbor_cell["colour"] if neighbor_cell else 0,
            }

        ghost_relative = np.array(
            [
                self.ghost_pos[0] - self.agent_pos[0],
                self.ghost_pos[1] - self.agent_pos[1],
            ],
            dtype=np.int32,
        )

        return {
            "current_cell": current_obs,
            "neighbors": neighbors_obs,
            "ghost_relative_pos": ghost_relative,
            "ghost_distance": self._calculate_shortest_path_distance(
                self.agent_pos, self.ghost_pos
            ),
        }

    def _calculate_shortest_path_distance(
        self, start_pos: List[int], end_pos: List[int]
    ) -> int:
        """Calculates the shortest path distance between two positions using BFS.

        Args:
            start_pos (list): Starting [row, col].
            end_pos (list): Target [row, col].

        Returns:
            int: The shortest path distance.
        """
        if start_pos == end_pos:
            return 0

        from collections import deque

        queue = deque([(tuple(start_pos), 0)])
        visited = {tuple(start_pos)}

        while queue:
            (r, c), dist = queue.popleft()

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    if self._is_move_valid([r, c], [nr, nc]):
                        if (nr, nc) == tuple(end_pos):
                            return dist + 1
                        if (nr, nc) not in visited:
                            visited.add((nr, nc))
                            queue.append(((nr, nc), dist + 1))

        return -1  # Should not happen in this grid

    def render(self) -> Optional[np.ndarray]:
        """Renders the environment.

        Returns:
            np.ndarray, optional: RGB array if render_mode is "rgb_array".
        """
        if self.renderer:
            logger.debug(
                f"Rendering: Agent at {self.agent_pos}, Ghost at {self.ghost_pos}, Turn {self.current_turn}"
            )
            return self.renderer.render(
                agent_pos=self.agent_pos,
                ghost_pos=self.ghost_pos,
                grid=self.grid,
                walls_horizontal=self.walls_horizontal,
                walls_vertical=self.walls_vertical,
                step_count=self.step_count,
                current_turn=self.current_turn,
                info=self.info,
                use_ghost=self.use_ghost,
            )
        return None

    def close(self):
        """Cleans up resources."""
        if self.renderer:
            self.renderer.close()

    def set_goal(self, pos: Tuple[int, int]):
        """Dynamically updates the goal position.

        Args:
            pos (Tuple[int, int]): The new goal position [row, col].
        """
        # Clear previous goals
        for r in range(self.rows):
            for c in range(self.cols):
                self.grid[r, c]["is_goal"] = False

        # Set new goal
        self.grid[pos[0], pos[1]]["is_goal"] = True
        if "reached_goal" in self.info:
            del self.info["reached_goal"]
        logger.debug(f"Goal set to: {pos}")

    def get_grid_description(self) -> str:
        """Returns a natural language description of the grid and its contents.

        Returns:
            str: Description of the grid.
        """
        description = "Der Grid hat 4 Zeilen (0-3) und 5 Spalten (0-4).\n"
        color_map = {0: "weiß", 1: "rot", 2: "grün"}
        item_mapping = {
            "one_note": "klassische Musik und Klaviermusik",
            "two_notes": "Rockmusik",
            "flower": "eine Blume",
            "two_flowers": "zwei Blumen",
            "dog": "ein Bild eines Hundes",
        }

        for r in range(self.rows):
            for c in range(self.cols):
                cell = self.grid[r, c]

                # Map items to natural language descriptions
                mapped_items = []
                for item in cell["items"]:
                    if item in item_mapping:
                        mapped_items.append(item_mapping[item])
                    else:
                        mapped_items.append(item)

                items_str = ", ".join(mapped_items) if mapped_items else "keine"
                desc = f"Feld ({r}, {c}): Farbe {color_map[cell['colour']]}, Gegenstände: {items_str}"
                if cell["text"]:
                    desc += f", Text: '{cell['text']}'"

                description += desc + "\n"
        return description
