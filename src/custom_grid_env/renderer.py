"""Pygame-based renderer for the CustomGrid environment."""

import pygame
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from .logger import get_logger

logger = get_logger(__name__)


class PygameRenderer:
    """Renderer for the CustomGrid environment using Pygame."""

    def __init__(
        self, rows: int, cols: int, render_mode: str = "human", render_fps: int = 4
    ):
        """Initializes the PygameRenderer.

        Args:
            rows (int): Number of rows in the grid.
            cols (int): Number of columns in the grid.
            render_mode (str): Rendering mode ("human" or "rgb_array").
            render_fps (int): Frames per second for rendering.
        """
        self.rows = rows
        self.cols = cols
        self.render_mode = render_mode
        self.render_fps = render_fps

        # Load CNN model if available

        # Pygame setup constants
        self.cell_size = 100
        self.wall_thickness = 6
        self.window_width = self.cols * self.cell_size
        self.window_height = self.rows * self.cell_size + 185

        self.screen = None
        self.clock = None
        self.font = None
        self.small_font = None

        self.colors = {
            "white": (255, 255, 255),
            "black": (0, 0, 0),
            "red": (255, 100, 100),
            "red_dark": (200, 50, 50),
            "green": (100, 200, 100),
            "green_dark": (50, 150, 50),
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
            self._draw_crosshatch(
                surface,
                (
                    x + margin,
                    y + margin,
                    self.cell_size - 2 * margin,
                    self.cell_size - 2 * margin,
                ),
                self.colors["green"],
            )

        if cell["text"]:
            text = self.font.render(cell["text"], True, self.colors["black"])
            text_rect = text.get_rect(
                center=(x + self.cell_size // 2, y + self.cell_size // 2)
            )
            surface.blit(text, text_rect)

        if cell["items"] and not cell["is_goal"] and not cell["is_start"]:
            item_y_offset = 0
            note_offset = 0
            for item in cell["items"]:
                if "dog" in item:
                    self._draw_dog(
                        x + self.cell_size // 2,
                        y + self.cell_size // 2 + item_y_offset,
                        surface=surface,
                    )
                    item_y_offset += 20
                elif "flower" in item:
                    self._draw_flower(
                        x + self.cell_size // 2,
                        y + self.cell_size // 2 + item_y_offset,
                        surface=surface,
                    )
                    item_y_offset += 20
                elif "one_note" in item:
                    self._draw_note(
                        x + self.cell_size - 20,
                        y + 20 + note_offset,
                        single=True,
                        surface=surface,
                    )
                    note_offset += 25
                elif "two_notes" in item:
                    self._draw_note(
                        x + self.cell_size - 25,
                        y + 20 + note_offset,
                        single=False,
                        surface=surface,
                    )
                    note_offset += 25

    def _draw_dog(self, cx: int, cy: int, surface: Optional[pygame.Surface] = None):
        """Draws a simple dog icon."""
        if surface is None:
            surface = self.screen
        pygame.draw.ellipse(
            surface, self.colors["dark_gray"], (cx - 20, cy - 10, 40, 25)
        )
        pygame.draw.circle(surface, self.colors["dark_gray"], (cx - 15, cy - 15), 12)
        pygame.draw.ellipse(
            surface, self.colors["dark_gray"], (cx - 28, cy - 25, 10, 15)
        )
        pygame.draw.ellipse(
            surface, self.colors["dark_gray"], (cx - 12, cy - 25, 10, 15)
        )
        pygame.draw.circle(surface, self.colors["white"], (cx - 18, cy - 17), 3)
        pygame.draw.circle(surface, self.colors["white"], (cx - 12, cy - 17), 3)
        pygame.draw.arc(
            surface, self.colors["dark_gray"], (cx + 10, cy - 20, 20, 25), 0, 2, 3
        )

    def _draw_flower(self, cx: int, cy: int, surface: Optional[pygame.Surface] = None):
        """Draws a simple flower icon."""
        if surface is None:
            surface = self.screen
        petal_color = self.colors["white"]
        for angle in range(0, 360, 60):
            rad = np.radians(angle)
            px = cx + int(15 * np.cos(rad))
            py = cy + int(15 * np.sin(rad))
            pygame.draw.circle(surface, petal_color, (px, py), 10)
            pygame.draw.circle(surface, self.colors["dark_gray"], (px, py), 10, 1)
        pygame.draw.circle(surface, self.colors["yellow"], (cx, cy), 8)
        pygame.draw.circle(surface, self.colors["orange"], (cx, cy), 8, 2)

    def _draw_note(
        self,
        cx: int,
        cy: int,
        single: bool = True,
        surface: Optional[pygame.Surface] = None,
    ):
        """Draws musical note(s)."""
        if surface is None:
            surface = self.screen
        if single:
            pygame.draw.ellipse(surface, self.colors["black"], (cx - 8, cy, 12, 10))
            pygame.draw.line(
                surface,
                self.colors["black"],
                (cx + 3, cy + 5),
                (cx + 3, cy - 25),
                3,
            )
            pygame.draw.arc(
                surface, self.colors["black"], (cx, cy - 30, 15, 15), 3.5, 6, 3
            )
        else:
            pygame.draw.ellipse(surface, self.colors["black"], (cx - 18, cy, 12, 10))
            pygame.draw.ellipse(surface, self.colors["black"], (cx + 2, cy, 12, 10))
            pygame.draw.line(
                surface,
                self.colors["black"],
                (cx - 7, cy + 5),
                (cx - 7, cy - 20),
                3,
            )
            pygame.draw.line(
                surface,
                self.colors["black"],
                (cx + 13, cy + 5),
                (cx + 13, cy - 20),
                3,
            )
            pygame.draw.line(
                surface,
                self.colors["black"],
                (cx - 7, cy - 20),
                (cx + 13, cy - 20),
                3,
            )

    def _draw_agent(self, row: int, col: int):
        """Draws the agent."""
        x = col * self.cell_size + self.cell_size // 2
        y = row * self.cell_size + self.cell_size // 2
        pygame.draw.rect(
            self.screen, self.colors["gray"], (x - 25, y - 20, 50, 45), border_radius=5
        )
        pygame.draw.rect(
            self.screen,
            self.colors["dark_gray"],
            (x - 25, y - 20, 50, 45),
            2,
            border_radius=5,
        )
        pygame.draw.line(
            self.screen, self.colors["dark_gray"], (x, y - 20), (x, y - 35), 3
        )
        pygame.draw.circle(self.screen, self.colors["red"], (x, y - 35), 5)
        pygame.draw.rect(
            self.screen, self.colors["cyan"], (x - 18, y - 12, 36, 20), border_radius=3
        )
        pygame.draw.circle(self.screen, self.colors["dark_gray"], (x - 8, y - 2), 5)
        pygame.draw.circle(self.screen, self.colors["dark_gray"], (x + 8, y - 2), 5)
        gps_text = self.small_font.render("GPS", True, self.colors["dark_gray"])
        gps_rect = gps_text.get_rect(center=(x, y + 15))
        self.screen.blit(gps_text, gps_rect)
        pygame.draw.circle(self.screen, self.colors["dark_gray"], (x - 18, y + 28), 8)
        pygame.draw.circle(self.screen, self.colors["dark_gray"], (x + 18, y + 28), 8)

    def _draw_ghost(self, row: int, col: int):
        """Draws the ghost."""
        x = col * self.cell_size + self.cell_size // 2
        y = row * self.cell_size + self.cell_size // 2
        color = self.colors["cyan"]
        pygame.draw.circle(self.screen, color, (x, y - 5), 25)
        pygame.draw.rect(self.screen, color, (x - 25, y - 5, 50, 30))
        for i in range(5):
            wave_x = x - 20 + i * 10
            pygame.draw.circle(self.screen, color, (wave_x, y + 25), 5)
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
    ) -> Optional[np.ndarray]:
        """Renders the environment state."""
        self._init_pygame()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return None

        self.screen.fill(self.colors["white"])
        self._draw_grid_lines()
        for row in range(self.rows):
            for col in range(self.cols):
                self._draw_cell(row, col, grid[row, col])
        self._draw_walls(walls_horizontal, walls_vertical)
        if agent_pos != ghost_pos:
            self._draw_ghost(ghost_pos[0], ghost_pos[1])
        self._draw_agent(agent_pos[0], agent_pos[1])
        if agent_pos == ghost_pos:
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
