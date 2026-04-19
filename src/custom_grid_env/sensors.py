"""Sensor implementations for the CustomGrid environment."""

import os
import numpy as np
import pygame
import tensorflow as tf
from typing import Dict, Any, Optional, Tuple
from .logger import get_logger

logger = get_logger(__name__)


class VisionSensor:
    """Handles CNN-based vision for item classification."""

    def __init__(self, model_path: Optional[str] = None):
        """Initializes the VisionSensor.

        Args:
            model_path (str, optional): Path to the trained .keras model.
        """
        if model_path is None:
            # Default path relative to this file
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, "cnn_tutorial", "model.keras")

        self.model = None
        if os.path.exists(model_path):
            try:
                self.model = tf.keras.models.load_model(model_path)
                logger.info(f"Loaded CNN model from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load CNN model: {e}")
        else:
            logger.warning(f"CNN model not found at {model_path}. Vision disabled.")

        self.class_names = ["dog", "flower", "background"]

        # Define colors for drawing
        self.colors = {
            "white": (255, 255, 255),
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "yellow": (255, 255, 0),
            "orange": (255, 165, 0),
            "dark_gray": (64, 64, 64),
            "black": (0, 0, 0),
        }

    def predict(self, cell: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Returns the CNN prediction for a given cell.

        Args:
            cell (dict): The cell dictionary from the environment grid.

        Returns:
            dict, optional: Dictionary containing 'prediction' (class, prob) and 'probs' (list of floats).
        """
        if self.model is None:
            return None

        # Create a 64x64 surface to draw the cell content
        # Ensure pygame is initialized for surface creation (headless is fine)
        if not pygame.get_init():
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            pygame.init()

        temp_surface = pygame.Surface((64, 64))
        self._draw_cell_for_cnn(cell, temp_surface)

        # Convert surface to numpy array (RGB)
        img_array = pygame.surfarray.array3d(temp_surface)
        # Transpose from (W, H, C) to (H, W, C)
        img_array = np.transpose(img_array, (1, 0, 2))
        # Normalize
        img_array = img_array.astype(np.float32) / 255.0
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = self.model.predict(img_array, verbose=0)
        probs = predictions[0].tolist()
        class_idx = np.argmax(predictions[0])
        probability = float(predictions[0][class_idx])

        return {
            "prediction": (self.class_names[class_idx], probability),
            "probs": probs,
        }

    def _draw_cell_for_cnn(self, cell: Dict[str, Any], surface: pygame.Surface):
        """Draws cell content on a 64x64 surface for CNN input.

        Args:
            cell (Dict[str, Any]): The cell metadata.
            surface (pygame.Surface): The surface to draw on.
        """
        surface.fill(self.colors["white"])
        cell_size = 64
        margin = 4
        cx, cy = cell_size // 2, cell_size // 2

        # Draw background color (crosshatch)
        if cell["colour"] == 1:  # Red
            self._draw_crosshatch(
                surface,
                (margin, margin, cell_size - 2 * margin, cell_size - 2 * margin),
                self.colors["red"],
            )
        elif cell["colour"] == 2:  # Green
            self._draw_crosshatch(
                surface,
                (margin, margin, cell_size - 2 * margin, cell_size - 2 * margin),
                self.colors["green"],
            )

        # Draw items
        if "dog" in cell["items"]:
            self._draw_dog(cx, cy, surface)
        elif "flower" in cell["items"]:
            self._draw_flower(cx, cy, surface)
        elif "notes" in cell["items"]:
            self._draw_note(cx, cy, surface)

    def _draw_crosshatch(
        self,
        surface: pygame.Surface,
        rect: Tuple[int, int, int, int],
        color: Tuple[int, int, int],
    ):
        """Draws a crosshatch pattern on the surface.

        Args:
            surface (pygame.Surface): The surface to draw on.
            rect (Tuple[int, int, int, int]): The rectangle (x, y, w, h) to fill.
            color (Tuple[int, int, int]): The RGB color of the lines.
        """
        x, y, w, h = rect
        for i in range(0, w + h, 8):
            pygame.draw.line(
                surface,
                color,
                (x + max(0, i - h), y + min(i, h)),
                (x + min(i, w), y + max(0, i - w)),
                1,
            )

    def _draw_dog(self, cx: int, cy: int, surface: pygame.Surface):
        """Draws a dog symbol on the surface.

        Args:
            cx (int): X-coordinate of the center.
            cy (int): Y-coordinate of the center.
            surface (pygame.Surface): The surface to draw on.
        """
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

    def _draw_flower(self, cx: int, cy: int, surface: pygame.Surface):
        """Draws a flower symbol on the surface.

        Args:
            cx (int): X-coordinate of the center.
            cy (int): Y-coordinate of the center.
            surface (pygame.Surface): The surface to draw on.
        """
        petal_color = self.colors["white"]
        for angle in range(0, 360, 60):
            rad = np.radians(angle)
            px = cx + int(15 * np.cos(rad))
            py = cy + int(15 * np.sin(rad))
            pygame.draw.circle(surface, petal_color, (px, py), 10)
            pygame.draw.circle(surface, self.colors["dark_gray"], (px, py), 10, 1)
        pygame.draw.circle(surface, self.colors["yellow"], (cx, cy), 8)
        pygame.draw.circle(surface, self.colors["orange"], (cx, cy), 8, 2)

    def _draw_note(self, cx: int, cy: int, surface: pygame.Surface):
        """Draws a musical note symbol on the surface.

        Args:
            cx (int): X-coordinate of the center.
            cy (int): Y-coordinate of the center.
            surface (pygame.Surface): The surface to draw on.
        """
        pygame.draw.ellipse(surface, self.colors["black"], (cx - 8, cy, 12, 10))
        pygame.draw.line(
            surface, self.colors["black"], (cx + 3, cy + 5), (cx + 3, cy - 25), 3
        )
        pygame.draw.line(
            surface, self.colors["black"], (cx + 3, cy - 25), (cx + 13, cy - 20), 3
        )
