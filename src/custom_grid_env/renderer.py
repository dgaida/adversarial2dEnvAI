"""Pygame-based renderer for the CustomGrid environment."""

import pygame
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from .logger import get_logger

logger = get_logger(__name__)


class PygameRenderer:
    """Renderer for the CustomGrid environment using Pygame."""

    def __init__(
        self,
        rows: int,
        cols: int,
        cell_size: int = 100,
        render_mode: str = "human",
        render_fps: int = 4,
    ):
        """Initializes the PygameRenderer.

        Args:
            rows (int): Number of rows in the grid.
            cols (int): Number of columns in the grid.
            cell_size (int): Size of each cell in pixels. Defaults to 100.
            render_mode (str): The mode to render with. Defaults to "human".
            render_fps (int): Rendering frames per second. Defaults to 4.
        """
        self.rows = rows
        self.cols = cols
        self.cell_size = cell_size
        self.render_mode = render_mode
        self.render_fps = render_fps
        self.window_width = cols * cell_size
        self.window_height = rows * cell_size + 185  # Extra space for info
        self.wall_thickness = 6

        self.screen = None
        self.clock = None
        self.font = None
        self.small_font = None

        self.colors = {
            "white": (255, 255, 255),
            "black": (0, 0, 0),
            "red": (255, 100, 100),
            "green": (100, 255, 100),
            "yellow": (255, 220, 100),
            "blue": (100, 150, 255),
            "purple": (180, 100, 220),
            "cyan": (100, 220, 220),
            "gray": (200, 200, 200),
            "dark_gray": (80, 80, 80),
            "orange": (255, 180, 100),
        }

    def _init_pygame(self):
        """Initializes Pygame if not already initialized."""
        if self.screen is None:
            pygame.init()
            pygame.display.set_caption("Custom Grid Environment")
            self.screen = pygame.display.set_mode(
                (self.window_width, self.window_height)
            )
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)
            self.small_font = pygame.font.Font(None, 24)

    def _draw_crosshatch(
        self,
        surface: pygame.Surface,
        rect: Tuple[int, int, int, int],
        color: Tuple[int, int, int],
        line_spacing: int = 8,
    ):
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

    def _draw_cell(
        self,
        row: int,
        col: int,
        cell: Dict[str, Any],
        surface: Optional[pygame.Surface] = None,
        agent_value: Optional[float] = None,
        ghost_value: Optional[float] = None,
    ):
        """Draws a single cell with its contents."""
        if surface is None:
            surface = self.screen

        # If drawing on a custom surface (e.g. for CNN), we might want to center it
        # but for now we keep the same logic as before.
        # However, for CNN we need exactly 64x64.
        # Let's adjust x, y if surface is small.
        surf_w, surf_h = surface.get_size()
        if surf_w == self.cell_size and surf_h == self.cell_size:
            x, y = 0, 0
        else:
            x = col * self.cell_size
            y = row * self.cell_size

        margin = 4

        pygame.draw.rect(
            surface,
            self.colors["white"],
            (
                x + margin,
                y + margin,
                self.cell_size - 2 * margin,
                self.cell_size - 2 * margin,
            ),
        )

        # Draw agent and ghost values if provided
        if agent_value is not None:
            val_text = self.small_font.render(
                f"{agent_value:.0f}", True, self.colors["blue"]
            )
            surface.blit(val_text, (x + 8, y + self.cell_size - 22))

        if ghost_value is not None:
            val_text = self.small_font.render(
                f"{ghost_value:.0f}", True, self.colors["purple"]
            )
            val_rect = val_text.get_rect()
            surface.blit(
                val_text,
                (x + self.cell_size - val_rect.width - 8, y + self.cell_size - 22),
            )

        if cell["colour"] == 1:
            self._draw_crosshatch(
                surface,
                (
                    x + margin,
                    y + margin,
                    self.cell_size - 2 * margin,
                    self.cell_size - 2 * margin,
                ),
                self.colors["red"],
            )
        elif cell["colour"] == 2:
            pygame.draw.rect(
                surface,
                self.colors["green"],
                (
                    x + margin,
                    y + margin,
                    self.cell_size - 2 * margin,
                    self.cell_size - 2 * margin,
                ),
            )

        if "dog" in cell["items"]:
            self._draw_dog(surface, x + 50, y + 40)
        if "flower" in cell["items"]:
            self._draw_flower(surface, x + 70, y + 60)
        if "one_note" in cell["items"]:
            self._draw_music_note(surface, x + 20, y + 70, 1)
        if "two_notes" in cell["items"]:
            self._draw_music_note(surface, x + 20, y + 70, 2)

        if cell["text"]:
            text_surf = self.small_font.render(cell["text"], True, self.colors["black"])
            text_rect = text_surf.get_rect(center=(x + 50, y + 90))
            surface.blit(text_surf, text_rect)

    def _draw_dog(self, surface, x, y):
        """Draws a simple dog icon."""
        color = (139, 69, 19)  # Brown
        pygame.draw.ellipse(surface, color, (x - 25, y - 15, 50, 30))  # Body
        pygame.draw.circle(surface, color, (x + 15, y - 10), 12)  # Head
        pygame.draw.ellipse(surface, color, (x + 22, y - 15, 8, 15))  # Ear
        pygame.draw.line(surface, color, (x - 20, y + 5), (x - 20, y + 20), 4)  # Leg
        pygame.draw.line(surface, color, (x + 10, y + 5), (x + 10, y + 20), 4)  # Leg

    def _draw_flower(self, surface, x, y):
        """Draws a simple flower icon."""
        pygame.draw.circle(surface, (255, 255, 0), (x, y), 8)  # Center
        for i in range(5):
            angle = i * (2 * np.pi / 5)
            px = x + 12 * np.cos(angle)
            py = y + 12 * np.sin(angle)
            pygame.draw.circle(surface, (255, 105, 180), (int(px), int(py)), 8)

    def _draw_music_note(self, surface, x, y, count=1):
        """Draws simple music note icons."""
        color = self.colors["black"]
        for i in range(count):
            nx = x + i * 15
            pygame.draw.circle(surface, color, (nx, y), 6)
            pygame.draw.line(surface, color, (nx + 4, y), (nx + 4, y - 15), 2)
            pygame.draw.line(surface, color, (nx + 4, y - 15), (nx + 12, y - 10), 2)

    def _draw_agent(self, row, col):
        """Draws the agent icon."""
        x = col * self.cell_size + self.cell_size // 2
        y = row * self.cell_size + self.cell_size // 2
        color = self.colors["blue"]
        pygame.draw.circle(self.screen, color, (x, y - 10), 20)  # Head
        pygame.draw.rect(self.screen, color, (x - 15, y + 10, 30, 30))  # Body

    def _draw_ghost(self, row, col):
        """Draws the ghost icon."""
        x = col * self.cell_size + self.cell_size // 2
        y = row * self.cell_size + self.cell_size // 2
        color = self.colors["cyan"]
        # Ghost body
        pygame.draw.circle(self.screen, color, (x, y - 5), 25)
        pygame.draw.rect(self.screen, color, (x - 25, y - 5, 50, 30))
        for i in range(5):
            wave_x = x - 20 + i * 10
            pygame.draw.circle(self.screen, color, (wave_x, y + 25), 5)
        # Eyes
        pygame.draw.circle(self.screen, self.colors["white"], (x - 10, y - 8), 8)
        pygame.draw.circle(self.screen, self.colors["white"], (x + 10, y - 8), 8)
        pygame.draw.circle(self.screen, self.colors["dark_gray"], (x - 8, y - 6), 4)
        pygame.draw.circle(self.screen, self.colors["dark_gray"], (x + 12, y - 6), 4)

    def _draw_walls(self, walls_horizontal: np.ndarray, walls_vertical: np.ndarray):
        """Draws all walls."""
        wall_color = self.colors["black"]
        for row in range(self.rows):
            for col in range(self.cols - 1):
                if walls_vertical[row, col]:
                    x = (col + 1) * self.cell_size
                    y = row * self.cell_size
                    pygame.draw.rect(
                        self.screen,
                        wall_color,
                        (
                            x - self.wall_thickness // 2,
                            y,
                            self.wall_thickness,
                            self.cell_size,
                        ),
                    )

        for row in range(self.rows - 1):
            for col in range(self.cols):
                if walls_horizontal[row, col]:
                    x = col * self.cell_size
                    y = (row + 1) * self.cell_size
                    pygame.draw.rect(
                        self.screen,
                        wall_color,
                        (
                            x,
                            y - self.wall_thickness // 2,
                            self.cell_size,
                            self.wall_thickness,
                        ),
                    )

    def _draw_grid_lines(self):
        """Draws the grid lines."""
        line_color = self.colors["gray"]
        for col in range(self.cols + 1):
            x = col * self.cell_size
            pygame.draw.line(
                self.screen, line_color, (x, 0), (x, self.rows * self.cell_size), 1
            )
        for row in range(self.rows + 1):
            y = row * self.cell_size
            pygame.draw.line(self.screen, line_color, (0, y), (self.window_width, y), 1)

    def _draw_info_panel(
        self,
        agent_pos: List[int],
        ghost_pos: List[int],
        step_count: int,
        current_turn: int,
        grid: np.ndarray,
        info: Dict[str, Any],
    ):
        """Draws the information panel at the bottom."""
        panel_y = self.rows * self.cell_size
        pygame.draw.rect(
            self.screen, self.colors["dark_gray"], (0, panel_y, self.window_width, 185)
        )

        # Column offsets
        col1_x = 10
        col2_x = 180
        col3_x = 330

        # Row 1: Step and Turn
        step_text = self.font.render(f"Step: {step_count}", True, self.colors["white"])
        self.screen.blit(step_text, (col1_x, panel_y + 10))

        turn_name = "Agent's Turn" if current_turn == 0 else "Ghost's Turn"
        turn_color = self.colors["yellow"] if current_turn == 0 else self.colors["cyan"]
        turn_text = self.font.render(turn_name, True, turn_color)
        self.screen.blit(turn_text, (col2_x, panel_y + 10))

        # CNN Prediction (Moved to Col 3, Row 2/3)
        cnn_label = self.small_font.render("CNN:", True, self.colors["orange"])
        self.screen.blit(cnn_label, (col3_x, panel_y + 45))

        prediction = info.get("cnn_prediction")
        if prediction:
            class_name, prob = prediction
            pred_text = self.small_font.render(
                f"{class_name} ({prob*100:.1f}%)",
                True,
                self.colors["orange"],
            )
            self.screen.blit(pred_text, (col3_x, panel_y + 70))

        # Row 2: Agent Pos, Cell Info
        pos_text = self.small_font.render(
            f"Agent: ({agent_pos[0]}, {agent_pos[1]})",
            True,
            self.colors["white"],
        )
        self.screen.blit(pos_text, (col1_x, panel_y + 45))

        current_cell = grid[agent_pos[0], agent_pos[1]]
        colour_name = (
            "None"
            if current_cell["colour"] == 0
            else "Red" if current_cell["colour"] == 1 else "Green"
        )
        cell_text = self.small_font.render(
            f"Color: {colour_name}", True, self.colors["white"]
        )
        self.screen.blit(cell_text, (col2_x, panel_y + 45))

        # Row 3: Ghost Pos, Items
        ghost_text = self.small_font.render(
            f"Ghost: ({ghost_pos[0]}, {ghost_pos[1]})",
            True,
            self.colors["cyan"],
        )
        self.screen.blit(ghost_text, (col1_x, panel_y + 70))

        items_str = (
            ", ".join(current_cell["items"]) if current_cell["items"] else "None"
        )
        # Truncate items if too long
        if len(items_str) > 15:
            items_str = items_str[:12] + "..."
        items_text = self.small_font.render(
            f"Items: {items_str}", True, self.colors["white"]
        )
        self.screen.blit(items_text, (col2_x, panel_y + 70))

        # Row 4: Distance, Goal, Sensor
        distance = info.get(
            "ghost_distance",
            abs(agent_pos[0] - ghost_pos[0]) + abs(agent_pos[1] - ghost_pos[1]),
        )
        dist_text = self.small_font.render(
            f"Dist: {distance}", True, self.colors["yellow"]
        )
        self.screen.blit(dist_text, (col1_x, panel_y + 95))

        goal_text = self.small_font.render(
            f"Goal: {'Yes' if current_cell['is_goal'] else 'No'}",
            True,
            self.colors["yellow"] if current_cell["is_goal"] else self.colors["white"],
        )
        self.screen.blit(goal_text, (col2_x, panel_y + 95))

        color_measurement = info.get("color_measurement")
        if color_measurement is not None:
            color_names = ["White", "Red", "Green"]
            color_text = self.small_font.render(
                f"Sensor: {color_names[color_measurement]}",
                True,
                self.colors["white"],
            )
            self.screen.blit(color_text, (col3_x, panel_y + 95))

        # Row 5 & 6: Actions and Est. Pos
        est_pos_data = info.get("estimated_pos", {})
        est_pos = (
            est_pos_data.get("cell_pos") if isinstance(est_pos_data, dict) else None
        )
        if est_pos is not None:
            est_text = self.small_font.render(
                f"Est. Pos: ({est_pos[0]}, {est_pos[1]})",
                True,
                self.colors["orange"],
            )
            self.screen.blit(est_text, (col3_x, panel_y + 125))

        # Row 5 & 6: Actions (Occupies bottom)
        intended_action = self.small_font.render(
            f"Intended: {info.get('intended_action', '')}",
            True,
            self.colors["white"],
        )
        self.screen.blit(intended_action, (col1_x, panel_y + 125))

        actual_action_str = info.get("actual_action", "")
        actual_action = self.small_font.render(
            f"Actual: {actual_action_str}",
            True,
            self.colors["white"],
        )
        self.screen.blit(actual_action, (col1_x, panel_y + 150))

    def render(
        self,
        agent_pos: List[int],
        ghost_pos: List[int],
        grid: np.ndarray,
        walls_horizontal: np.ndarray,
        walls_vertical: np.ndarray,
        step_count: int,
        current_turn: int,
        info: Dict[str, Any],
        use_ghost: bool = True,
    ) -> Optional[np.ndarray]:
        """Renders the environment state."""
        self._init_pygame()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return None

        agent_values = info.get("agent_values")
        ghost_values = info.get("ghost_values")

        self.screen.fill(self.colors["white"])
        self._draw_grid_lines()
        for row in range(self.rows):
            for col in range(self.cols):
                av = agent_values[row, col] if agent_values is not None else None
                gv = ghost_values[row, col] if ghost_values is not None else None
                self._draw_cell(
                    row, col, grid[row, col], agent_value=av, ghost_value=gv
                )
        self._draw_walls(walls_horizontal, walls_vertical)
        if use_ghost and agent_pos != ghost_pos:
            self._draw_ghost(ghost_pos[0], ghost_pos[1])
        self._draw_agent(agent_pos[0], agent_pos[1])
        if use_ghost and agent_pos == ghost_pos:
            x = agent_pos[1] * self.cell_size + self.cell_size // 2
            y = agent_pos[0] * self.cell_size + self.cell_size // 2
            pygame.draw.circle(self.screen, self.colors["red"], (x, y), 45, 5)
            bang_text = self.font.render("!", True, self.colors["red"])
            self.screen.blit(bang_text, (x - 5, y - 15))
        # Draw particles if available
        if "particles" in info and info.get("show_particles", True):
            self._draw_particles(info["particles"])

        self._draw_info_panel(
            agent_pos, ghost_pos, step_count, current_turn, grid, info
        )
        pygame.draw.rect(
            self.screen,
            self.colors["black"],
            (0, 0, self.window_width, self.rows * self.cell_size),
            3,
        )
        pygame.display.flip()
        self.clock.tick(self.render_fps)

        if self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        return None

    def _draw_particles(self, particles: List[List[int]]):
        """Draws particles as tiny dots on the grid.

        Args:
            particles (list): List of [row, col] positions.
        """
        for row, col in particles:
            # Add some jitter to distribute particles within the cell
            jitter_x = np.random.randint(-40, 40)
            jitter_y = np.random.randint(-40, 40)
            x = col * self.cell_size + self.cell_size // 2 + jitter_x
            y = row * self.cell_size + self.cell_size // 2 + jitter_y
            pygame.draw.circle(self.screen, self.colors["black"], (x, y), 2)

    def close(self):
        """Cleans up resources."""
        if self.screen is not None:
            pygame.quit()
            self.screen = None
