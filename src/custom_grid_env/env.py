import gymnasium as gym
import numpy as np
import pygame
import sys
from typing import Optional, Tuple, Dict, Any, List


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

    def __init__(self, render_mode: str = "human", slip_probability: float = 0.2):
        """Initializes the CustomGridEnv.

        Args:
            render_mode (str): The mode to render with. Defaults to "human".
            slip_probability (float): Chance to move perpendicular to intended direction. Defaults to 0.2.
        """
        super().__init__()
        self.render_mode = render_mode
        self.slip_probability = slip_probability
        self.rows = 4
        self.cols = 5
        self.observation_space = gym.spaces.Dict({
            "current_cell": gym.spaces.Dict({
                "colour": gym.spaces.Discrete(3),  # 0=none, 1=red, 2=green
                "has_item": gym.spaces.MultiBinary(3),  # [dog, flower, notes]
                "is_goal": gym.spaces.Discrete(2),
                "text": gym.spaces.Text(max_length=10),
            }),
            "neighbors": gym.spaces.Dict({
                "up": gym.spaces.Dict({
                    "accessible": gym.spaces.Discrete(2),
                    "colour": gym.spaces.Discrete(3),
                }),
                "right": gym.spaces.Dict({
                    "accessible": gym.spaces.Discrete(2),
                    "colour": gym.spaces.Discrete(3),
                }),
                "down": gym.spaces.Dict({
                    "accessible": gym.spaces.Discrete(2),
                    "colour": gym.spaces.Discrete(3),
                }),
                "left": gym.spaces.Dict({
                    "accessible": gym.spaces.Discrete(2),
                    "colour": gym.spaces.Discrete(3),
                }),
            }),
            "ghost_relative_pos": gym.spaces.Box(low=-4, high=4, shape=(2,), dtype=np.int32),
        })

        self.action_space = gym.spaces.Discrete(4)  # 0: left, 1: down, 2: right, 3: up

        self.grid = np.empty((self.rows, self.cols), dtype=object)
        self._setup_grid()
        self._setup_walls()
        self.agent_pos = [0, 2]
        self.start_pos = [0, 2]
        self.ghost_pos = [0, 3]
        self.ghost_start_pos = [0, 3]
        self.step_count = 0
        self.current_turn = 0
        self.info = {}

        # Pygame setup
        self.cell_size = 100
        self.wall_thickness = 6
        self.window_width = self.cols * self.cell_size
        self.window_height = self.rows * self.cell_size + 145
        self.screen = None
        self.clock = None
        self.font = None
        self.small_font = None

        self.colors = {
            'white': (255, 255, 255),
            'black': (0, 0, 0),
            'red': (255, 100, 100),
            'red_dark': (200, 50, 50),
            'green': (100, 200, 100),
            'green_dark': (50, 150, 50),
            'yellow': (255, 220, 100),
            'blue': (100, 150, 255),
            'purple': (180, 100, 220),
            'cyan': (100, 220, 220),
            'gray': (200, 200, 200),
            'dark_gray': (80, 80, 80),
            'orange': (255, 180, 100),
        }

    def _setup_grid(self):
        """Sets up the initial grid contents."""
        self.grid[0, 0] = {'colour': 2, 'items': ['dog'], 'is_goal': False, 'is_start': False, 'text': ''}
        self.grid[0, 1] = {'colour': 1, 'items': [], 'is_goal': False, 'is_start': False, 'text': ''}
        self.grid[0, 2] = {'colour': 0, 'items': [], 'is_goal': False, 'is_start': True, 'text': 'Start'}
        self.grid[0, 3] = {'colour': 0, 'items': [], 'is_goal': False, 'is_start': False, 'text': ''}
        self.grid[0, 4] = {'colour': 0, 'items': ['dog', 'one_note'], 'is_goal': False, 'is_start': False, 'text': ''}

        self.grid[1, 0] = {'colour': 0, 'items': [], 'is_goal': False, 'is_start': False, 'text': ''}
        self.grid[1, 1] = {'colour': 2, 'items': [], 'is_goal': False, 'is_start': False, 'text': ''}
        self.grid[1, 2] = {'colour': 0, 'items': [], 'is_goal': False, 'is_start': False, 'text': ''}
        self.grid[1, 3] = {'colour': 0, 'items': ['flower'], 'is_goal': False, 'is_start': False, 'text': ''}
        self.grid[1, 4] = {'colour': 2, 'items': [], 'is_goal': False, 'is_start': False, 'text': ''}

        self.grid[2, 0] = {'colour': 1, 'items': [], 'is_goal': False, 'is_start': False, 'text': ''}
        self.grid[2, 1] = {'colour': 2, 'items': [], 'is_goal': False, 'is_start': False, 'text': ''}
        self.grid[2, 2] = {'colour': 0, 'items': ['one_note'], 'is_goal': False, 'is_start': False, 'text': ''}
        self.grid[2, 3] = {'colour': 1, 'items': [], 'is_goal': False, 'is_start': False, 'text': ''}
        self.grid[2, 4] = {'colour': 0, 'items': [], 'is_goal': False, 'is_start': False, 'text': ''}

        self.grid[3, 0] = {'colour': 1, 'items': [], 'is_goal': False, 'is_start': False, 'text': ''}
        self.grid[3, 1] = {'colour': 0, 'items': [], 'is_goal': True, 'is_start': False, 'text': 'Ziel'}
        self.grid[3, 2] = {'colour': 2, 'items': ['two_notes'], 'is_goal': False, 'is_start': False, 'text': ''}
        self.grid[3, 3] = {'colour': 0, 'items': ['flower'], 'is_goal': False, 'is_start': False, 'text': ''}
        self.grid[3, 4] = {'colour': 0, 'items': [], 'is_goal': True, 'is_start': False, 'text': 'Ziel'}

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

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
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
        self.info = {}
        return self._get_obs(), {"current_turn": "agent"}

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
            if action == 0:
                if col > 0 and self.walls_vertical[new_row, new_col]:
                    move_valid = False
            elif action == 1:
                if row < self.rows and self.walls_horizontal[new_row - 1, new_col]:
                    move_valid = False
            elif action == 2:
                if col < self.cols and self.walls_vertical[new_row, new_col - 1]:
                    move_valid = False
            elif action == 3:
                if row > 0 and self.walls_horizontal[new_row, new_col]:
                    move_valid = False

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
            "colour": current_cell['colour'],
            "has_item": np.array([
                1 if 'dog' in current_cell['items'] else 0,
                1 if 'flower' in current_cell['items'] else 0,
                1 if any(note in current_cell['items'] for note in ['one_note', 'two_notes']) else 0
            ], dtype=np.int8),
            "is_goal": 1 if current_cell['is_goal'] else 0,
        }

        neighbors_obs = {}
        for direction, (dr, dc) in [("up", (-1, 0)), ("right", (0, 1)), ("down", (1, 0)), ("left", (0, -1))]:
            neighbor_pos = [row + dr, col + dc]
            accessible = self._is_move_valid(self.ghost_pos, neighbor_pos)
            neighbor_cell = self._get_cell_info(row + dr, col + dc)
            neighbors_obs[direction] = {
                "accessible": 1 if accessible else 0,
                "colour": neighbor_cell['colour'] if neighbor_cell else 0,
            }

        agent_relative = np.array([
            self.agent_pos[0] - self.ghost_pos[0],
            self.agent_pos[1] - self.ghost_pos[1]
        ], dtype=np.int32)

        return {
            "current_cell": current_obs,
            "neighbors": neighbors_obs,
            "agent_relative_pos": agent_relative,
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
            "terminal_states": ["caught_by_ghost", "reached_goal"]
        }

    def calculate_reward(self, caught_by_ghost: bool = False) -> Tuple[float, bool, Dict[str, Any]]:
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
            return float(reward_structure["caught_by_ghost"]), True, {"caught_by_ghost": True}

        current_cell = self.grid[self.agent_pos[0], self.agent_pos[1]]

        if current_cell['is_goal']:
            return float(reward_structure["reached_goal"]), True, {"reached_goal": True}

        return float(reward), terminated, {}

    def _apply_slip(self, intended_action: int) -> Tuple[int, bool]:
        """Applies slip probability to potentially change the action.

        Args:
            intended_action (int): The action the agent intended to take.

        Returns:
            tuple: (actual_action, slipped)
        """
        if self.slip_probability <= 0:
            return intended_action, False

        perpendicular = {
            0: [1, 3],
            1: [0, 2],
            2: [1, 3],
            3: [0, 2],
        }

        if self.np_random.random() < self.slip_probability:
            actual_action = int(self.np_random.choice(perpendicular[intended_action]))
            return actual_action, True

        return intended_action, False

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Executes one step in the environment.

        Args:
            action (int): Action for current entity.

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        info = {}
        reward = 0.0
        terminated = False
        action_names = {0: "left", 1: "down", 2: "right", 3: "up"}

        if self.current_turn == 0:
            self.step_count += 1
            actual_action, slipped = self._apply_slip(action)
            self.agent_pos = self._move_entity(self.agent_pos, actual_action)

            info['slipped'] = slipped
            info['intended_action'] = action_names[action]
            info['actual_action'] = action_names[actual_action]

            current_cell = self.grid[self.agent_pos[0], self.agent_pos[1]]
            if self.agent_pos == self.ghost_pos:
                reward = float(self.get_reward_structure()["caught_by_ghost"])
                terminated = True
                info['caught_by_ghost'] = True
            elif current_cell['is_goal']:
                reward = float(self.get_reward_structure()["reached_goal"])
                terminated = True
                info['reached_goal'] = True
            else:
                reward = float(self.get_reward_structure()["step_penalty"])

            self.current_turn = 1
            info['current_turn'] = 'ghost'
            info['mover'] = 'agent'
            self.info = info

        else:
            self.move_ghost(action)
            if self.agent_pos == self.ghost_pos:
                reward = float(self.get_reward_structure()["caught_by_ghost"])
                terminated = True
                info['caught_by_ghost'] = True

            self.current_turn = 0
            info['current_turn'] = 'agent'
            info['mover'] = 'ghost'

        return self._get_obs(), float(reward), terminated, False, info

    def get_current_turn(self) -> str:
        """Returns whose turn it is.

        Returns:
            str: 'agent' or 'ghost'.
        """
        return 'agent' if self.current_turn == 0 else 'ghost'

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
            return not (from_row > 0 and self.walls_horizontal[from_row - 1, from_col])
        elif to_row > from_row:
            return not (from_row < self.rows and self.walls_horizontal[from_row, from_col])
        elif to_col < from_col:
            return not (from_col > 0 and self.walls_vertical[from_row, from_col - 1])
        elif to_col > from_col:
            return not (from_col < self.cols and self.walls_vertical[from_row, from_col])

        return True

    def _get_obs(self) -> Dict[str, Any]:
        """Returns detailed observation about current cell and neighbors.

        Returns:
            dict: Detailed observation.
        """
        row, col = self.agent_pos
        current_cell = self.grid[row, col]

        current_obs = {
            "colour": current_cell['colour'],
            "has_item": np.array([
                1 if 'dog' in current_cell['items'] else 0,
                1 if 'flower' in current_cell['items'] else 0,
                1 if any(note in current_cell['items'] for note in ['one_note', 'two_notes']) else 0
            ], dtype=np.int8),
            "is_goal": 1 if current_cell['is_goal'] else 0,
            "text": current_cell['text']
        }

        neighbors_obs = {}
        for direction, (dr, dc) in [("up", (-1, 0)), ("right", (0, 1)), ("down", (1, 0)), ("left", (0, -1))]:
            neighbor_pos = [row + dr, col + dc]
            accessible = self._is_move_valid(self.agent_pos, neighbor_pos)
            neighbor_cell = self._get_cell_info(row + dr, col + dc)
            neighbors_obs[direction] = {
                "accessible": 1 if accessible else 0,
                "colour": neighbor_cell['colour'] if neighbor_cell else 0,
            }

        ghost_relative = np.array([
            self.ghost_pos[0] - self.agent_pos[0],
            self.ghost_pos[1] - self.agent_pos[1]
        ], dtype=np.int32)

        return {
            "current_cell": current_obs,
            "neighbors": neighbors_obs,
            "ghost_relative_pos": ghost_relative,
        }

    def _init_pygame(self):
        """Initializes Pygame if not already initialized."""
        if self.screen is None:
            pygame.init()
            pygame.display.set_caption("Custom Grid Environment")
            self.screen = pygame.display.set_mode((self.window_width, self.window_height))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)
            self.small_font = pygame.font.Font(None, 24)

    def _draw_crosshatch(self, surface: pygame.Surface, rect: Tuple[int, int, int, int], color: Tuple[int, int, int], line_spacing: int = 8):
        """Draws a crosshatch pattern inside a rectangle."""
        x, y, w, h = rect
        for i in range(-h, w, line_spacing):
            start_x = max(x, x + i)
            start_y = max(y, y - i)
            end_x = min(x + w, x + i + h)
            end_y = min(y + h, y - i + w)
            if start_x < end_x:
                pygame.draw.line(surface, color, (start_x, start_y), (end_x, end_y), 2)

        for i in range(0, w + h, line_spacing):
            start_x = min(x + w, x + i)
            start_y = max(y, y + i - w)
            end_x = max(x, x + i - h)
            end_y = min(y + h, y + i)
            if start_x > end_x:
                pygame.draw.line(surface, color, (start_x, start_y), (end_x, end_y), 2)

    def _draw_cell(self, row: int, col: int):
        """Draws a single cell with its contents."""
        x = col * self.cell_size
        y = row * self.cell_size
        cell = self.grid[row, col]
        margin = 4

        pygame.draw.rect(self.screen, self.colors['white'],
                        (x + margin, y + margin, self.cell_size - 2*margin, self.cell_size - 2*margin))

        if cell['colour'] == 1:
            self._draw_crosshatch(self.screen,
                                 (x + margin, y + margin, self.cell_size - 2*margin, self.cell_size - 2*margin),
                                 self.colors['red'])
        elif cell['colour'] == 2:
            self._draw_crosshatch(self.screen,
                                 (x + margin, y + margin, self.cell_size - 2*margin, self.cell_size - 2*margin),
                                 self.colors['green'])

        if cell['text']:
            text = self.font.render(cell['text'], True, self.colors['black'])
            text_rect = text.get_rect(center=(x + self.cell_size // 2, y + self.cell_size // 2))
            self.screen.blit(text, text_rect)

        if cell['items'] and not cell['is_goal'] and not cell['is_start']:
            item_y_offset = 0
            note_offset = 0
            for item in cell['items']:
                if 'dog' in item:
                    self._draw_dog(x + self.cell_size // 2, y + self.cell_size // 2 + item_y_offset)
                    item_y_offset += 20
                elif 'flower' in item:
                    self._draw_flower(x + self.cell_size // 2, y + self.cell_size // 2 + item_y_offset)
                    item_y_offset += 20
                elif 'one_note' in item:
                    self._draw_note(x + self.cell_size - 20, y + 20 + note_offset, single=True)
                    note_offset += 25
                elif 'two_notes' in item:
                    self._draw_note(x + self.cell_size - 25, y + 20 + note_offset, single=False)
                    note_offset += 25

    def _draw_dog(self, cx: int, cy: int):
        """Draws a simple dog icon."""
        pygame.draw.ellipse(self.screen, self.colors['dark_gray'], (cx - 20, cy - 10, 40, 25))
        pygame.draw.circle(self.screen, self.colors['dark_gray'], (cx - 15, cy - 15), 12)
        pygame.draw.ellipse(self.screen, self.colors['dark_gray'], (cx - 28, cy - 25, 10, 15))
        pygame.draw.ellipse(self.screen, self.colors['dark_gray'], (cx - 12, cy - 25, 10, 15))
        pygame.draw.circle(self.screen, self.colors['white'], (cx - 18, cy - 17), 3)
        pygame.draw.circle(self.screen, self.colors['white'], (cx - 12, cy - 17), 3)
        pygame.draw.arc(self.screen, self.colors['dark_gray'], (cx + 10, cy - 20, 20, 25), 0, 2, 3)

    def _draw_flower(self, cx: int, cy: int):
        """Draws a simple flower icon."""
        petal_color = self.colors['white']
        for angle in range(0, 360, 60):
            rad = np.radians(angle)
            px = cx + int(15 * np.cos(rad))
            py = cy + int(15 * np.sin(rad))
            pygame.draw.circle(self.screen, petal_color, (px, py), 10)
            pygame.draw.circle(self.screen, self.colors['dark_gray'], (px, py), 10, 1)
        pygame.draw.circle(self.screen, self.colors['yellow'], (cx, cy), 8)
        pygame.draw.circle(self.screen, self.colors['orange'], (cx, cy), 8, 2)

    def _draw_note(self, cx: int, cy: int, single: bool = True):
        """Draws musical note(s)."""
        if single:
            pygame.draw.ellipse(self.screen, self.colors['black'], (cx - 8, cy, 12, 10))
            pygame.draw.line(self.screen, self.colors['black'], (cx + 3, cy + 5), (cx + 3, cy - 25), 3)
            pygame.draw.arc(self.screen, self.colors['black'], (cx, cy - 30, 15, 15), 3.5, 6, 3)
        else:
            pygame.draw.ellipse(self.screen, self.colors['black'], (cx - 18, cy, 12, 10))
            pygame.draw.ellipse(self.screen, self.colors['black'], (cx + 2, cy, 12, 10))
            pygame.draw.line(self.screen, self.colors['black'], (cx - 7, cy + 5), (cx - 7, cy - 20), 3)
            pygame.draw.line(self.screen, self.colors['black'], (cx + 13, cy + 5), (cx + 13, cy - 20), 3)
            pygame.draw.line(self.screen, self.colors['black'], (cx - 7, cy - 20), (cx + 13, cy - 20), 3)

    def _draw_agent(self, row: int, col: int):
        """Draws the agent."""
        x = col * self.cell_size + self.cell_size // 2
        y = row * self.cell_size + self.cell_size // 2
        pygame.draw.rect(self.screen, self.colors['gray'], (x - 25, y - 20, 50, 45), border_radius=5)
        pygame.draw.rect(self.screen, self.colors['dark_gray'], (x - 25, y - 20, 50, 45), 2, border_radius=5)
        pygame.draw.line(self.screen, self.colors['dark_gray'], (x, y - 20), (x, y - 35), 3)
        pygame.draw.circle(self.screen, self.colors['red'], (x, y - 35), 5)
        pygame.draw.rect(self.screen, self.colors['cyan'], (x - 18, y - 12, 36, 20), border_radius=3)
        pygame.draw.circle(self.screen, self.colors['dark_gray'], (x - 8, y - 2), 5)
        pygame.draw.circle(self.screen, self.colors['dark_gray'], (x + 8, y - 2), 5)
        gps_text = self.small_font.render("GPS", True, self.colors['dark_gray'])
        gps_rect = gps_text.get_rect(center=(x, y + 15))
        self.screen.blit(gps_text, gps_rect)
        pygame.draw.circle(self.screen, self.colors['dark_gray'], (x - 18, y + 28), 8)
        pygame.draw.circle(self.screen, self.colors['dark_gray'], (x + 18, y + 28), 8)

    def _draw_ghost(self, row: int, col: int):
        """Draws the ghost."""
        x = col * self.cell_size + self.cell_size // 2
        y = row * self.cell_size + self.cell_size // 2
        color = self.colors['cyan']
        pygame.draw.circle(self.screen, color, (x, y - 5), 25)
        pygame.draw.rect(self.screen, color, (x - 25, y - 5, 50, 30))
        for i in range(5):
            wave_x = x - 20 + i * 10
            pygame.draw.circle(self.screen, color, (wave_x, y + 25), 5)
        pygame.draw.circle(self.screen, self.colors['white'], (x - 10, y - 8), 8)
        pygame.draw.circle(self.screen, self.colors['white'], (x + 10, y - 8), 8)
        pygame.draw.circle(self.screen, self.colors['dark_gray'], (x - 8, y - 6), 4)
        pygame.draw.circle(self.screen, self.colors['dark_gray'], (x + 12, y - 6), 4)

    def _draw_walls(self):
        """Draws all walls."""
        wall_color = self.colors['black']
        for row in range(self.rows):
            for col in range(self.cols - 1):
                if self.walls_vertical[row, col]:
                    x = (col + 1) * self.cell_size
                    y = row * self.cell_size
                    pygame.draw.rect(self.screen, wall_color,
                                   (x - self.wall_thickness // 2, y, self.wall_thickness, self.cell_size))

        for row in range(self.rows - 1):
            for col in range(self.cols):
                if self.walls_horizontal[row, col]:
                    x = col * self.cell_size
                    y = (row + 1) * self.cell_size
                    pygame.draw.rect(self.screen, wall_color,
                                   (x, y - self.wall_thickness // 2, self.cell_size, self.wall_thickness))

    def _draw_grid_lines(self):
        """Draws the grid lines."""
        line_color = self.colors['gray']
        for col in range(self.cols + 1):
            x = col * self.cell_size
            pygame.draw.line(self.screen, line_color, (x, 0), (x, self.rows * self.cell_size), 1)
        for row in range(self.rows + 1):
            y = row * self.cell_size
            pygame.draw.line(self.screen, line_color, (0, y), (self.window_width, y), 1)

    def _draw_info_panel(self):
        """Draws the information panel at the bottom."""
        panel_y = self.rows * self.cell_size
        pygame.draw.rect(self.screen, self.colors['dark_gray'], (0, panel_y, self.window_width, 145))
        step_text = self.font.render(f"Step: {self.step_count}", True, self.colors['white'])
        self.screen.blit(step_text, (10, panel_y + 10))
        pos_text = self.small_font.render(f"Agent: ({self.agent_pos[0]}, {self.agent_pos[1]})", True, self.colors['white'])
        self.screen.blit(pos_text, (10, panel_y + 45))
        ghost_text = self.small_font.render(f"Ghost: ({self.ghost_pos[0]}, {self.ghost_pos[1]})", True, self.colors['cyan'])
        self.screen.blit(ghost_text, (10, panel_y + 70))
        distance = abs(self.agent_pos[0] - self.ghost_pos[0]) + abs(self.agent_pos[1] - self.ghost_pos[1])
        dist_text = self.small_font.render(f"Distance: {distance}", True, self.colors['yellow'])
        self.screen.blit(dist_text, (10, panel_y + 95))
        turn_name = "Agent's Turn" if self.current_turn == 0 else "Ghost's Turn"
        turn_color = self.colors['yellow'] if self.current_turn == 0 else self.colors['cyan']
        turn_text = self.font.render(turn_name, True, turn_color)
        self.screen.blit(turn_text, (200, panel_y + 10))
        current_cell = self.grid[self.agent_pos[0], self.agent_pos[1]]
        colour_name = 'None' if current_cell['colour'] == 0 else 'Red' if current_cell['colour'] == 1 else 'Green'
        cell_text = self.small_font.render(f"Cell colour: {colour_name}", True, self.colors['white'])
        self.screen.blit(cell_text, (200, panel_y + 45))
        items_str = ', '.join(current_cell['items']) if current_cell['items'] else 'None'
        items_text = self.small_font.render(f"Items: {items_str}", True, self.colors['white'])
        self.screen.blit(items_text, (200, panel_y + 70))
        goal_text = self.small_font.render(f"Goal: {'Yes' if current_cell['is_goal'] else 'No'}", True,
                                          self.colors['yellow'] if current_cell['is_goal'] else self.colors['white'])
        self.screen.blit(goal_text, (200, panel_y + 95))
        intended_action = self.small_font.render(f"Intended Action: {self.info.get('intended_action', '')}", True, self.colors['white'])
        self.screen.blit(intended_action, (10, panel_y + 120))
        actual_action = self.small_font.render(f"Actual Action: {self.info.get('actual_action', '')}", True, self.colors['white'])
        self.screen.blit(actual_action, (200, panel_y + 120))

    def render(self) -> Optional[np.ndarray]:
        """Renders the environment.

        Returns:
            np.ndarray, optional: RGB array if render_mode is "rgb_array".
        """
        self._init_pygame()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                sys.exit()

        self.screen.fill(self.colors['white'])
        self._draw_grid_lines()
        for row in range(self.rows):
            for col in range(self.cols):
                self._draw_cell(row, col)
        self._draw_walls()
        if self.agent_pos != self.ghost_pos:
            self._draw_ghost(self.ghost_pos[0], self.ghost_pos[1])
        self._draw_agent(self.agent_pos[0], self.agent_pos[1])
        if self.agent_pos == self.ghost_pos:
            x = self.agent_pos[1] * self.cell_size + self.cell_size // 2
            y = self.agent_pos[0] * self.cell_size + self.cell_size // 2
            pygame.draw.circle(self.screen, self.colors['red'], (x, y), 45, 5)
            bang_text = self.font.render("!", True, self.colors['red'])
            self.screen.blit(bang_text, (x - 5, y - 15))
        self._draw_info_panel()
        pygame.draw.rect(self.screen, self.colors['black'],
                        (0, 0, self.window_width, self.rows * self.cell_size), 3)
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

        if self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        return None

    def close(self):
        """Cleans up resources."""
        if self.screen is not None:
            pygame.quit()
            self.screen = None


gym.envs.registration.register(
    id="CustomGrid-v0",
    entry_point="custom_grid_env.env:CustomGridEnv",
)
