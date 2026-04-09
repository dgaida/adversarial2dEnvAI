import os
import pygame
import numpy as np
from pathlib import Path

# Set dummy video driver for headless environments
os.environ["SDL_VIDEODRIVER"] = "dummy"

def draw_crosshatch(surface, color, line_spacing=8):
    """Draws a crosshatch pattern on the surface."""
    w, h = surface.get_size()
    for i in range(-h, w, line_spacing):
        start_x = max(0, i)
        start_y = max(0, -i)
        end_x = min(w, i + h)
        end_y = min(h, -i + w)
        if start_x < end_x:
            pygame.draw.line(surface, color, (start_x, start_y), (end_x, end_y), 2)

    for i in range(0, w + h, line_spacing):
        start_x = min(w, i)
        start_y = max(0, i - w)
        end_x = max(0, i - h)
        end_y = min(h, i)
        if start_x > end_x:
            pygame.draw.line(surface, color, (start_x, start_y), (end_x, end_y), 2)

def draw_dog(surface, cx, cy):
    """Draws a simple dog icon."""
    dark_gray = (80, 80, 80)
    white = (255, 255, 255)

    # Body
    pygame.draw.ellipse(surface, dark_gray, (cx - 20, cy - 10, 40, 25))
    # Head
    pygame.draw.circle(surface, dark_gray, (cx - 15, cy - 15), 12)
    # Ears
    pygame.draw.ellipse(surface, dark_gray, (cx - 28, cy - 25, 10, 15))
    pygame.draw.ellipse(surface, dark_gray, (cx - 12, cy - 25, 10, 15))
    # Eyes
    pygame.draw.circle(surface, white, (cx - 18, cy - 17), 3)
    pygame.draw.circle(surface, white, (cx - 12, cy - 17), 3)
    # Tail
    pygame.draw.arc(surface, dark_gray, (cx + 10, cy - 20, 20, 25), 0, 2, 3)

def draw_flower(surface, cx, cy):
    """Draws a simple flower icon."""
    white = (255, 255, 255)
    dark_gray = (80, 80, 80)
    yellow = (255, 220, 100)
    orange = (255, 180, 100)

    petal_color = white
    for angle in range(0, 360, 60):
        rad = np.radians(angle)
        px = cx + int(15 * np.cos(rad))
        py = cy + int(15 * np.sin(rad))
        pygame.draw.circle(surface, petal_color, (px, py), 10)
        pygame.draw.circle(surface, dark_gray, (px, py), 10, 1)
    pygame.draw.circle(surface, yellow, (cx, cy), 8)
    pygame.draw.circle(surface, orange, (cx, cy), 8, 2)

def generate_data(output_dir="data", num_samples_per_class=100):
    """Generates images of dogs and flowers with different backgrounds."""
    pygame.init()
    img_size = 64

    # Define backgrounds
    backgrounds = [
        ("white", (255, 255, 255), None),
        ("red", (255, 255, 255), (255, 100, 100)), # white base, red crosshatch
        ("green", (255, 255, 255), (100, 200, 100)) # white base, green crosshatch
    ]

    classes = [("dog", draw_dog), ("flower", draw_flower)]

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    for class_name, draw_func in classes:
        class_dir = output_path / class_name
        class_dir.mkdir(exist_ok=True)
        print(f"Generating {num_samples_per_class} images for class: {class_name}")

        for i in range(num_samples_per_class):
            bg_name, bg_color, hatch_color = backgrounds[i % len(backgrounds)]

            surface = pygame.Surface((img_size, img_size))
            surface.fill(bg_color)

            if hatch_color:
                draw_crosshatch(surface, hatch_color)

            # Add some slight variation in position
            offset_x = np.random.randint(-2, 3)
            offset_y = np.random.randint(-2, 3)

            draw_func(surface, img_size // 2 + offset_x, img_size // 2 + offset_y)

            img_path = class_dir / f"{class_name}_{bg_name}_{i}.png"
            pygame.image.save(surface, str(img_path))

    pygame.quit()
    print(f"Data generation complete. Images saved to {output_dir}")

if __name__ == "__main__":
    generate_data(num_samples_per_class=300)
