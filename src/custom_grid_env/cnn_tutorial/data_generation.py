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


def draw_note(surface, cx, cy, single=True):
    """Draws musical note(s)."""
    black = (0, 0, 0)
    if single:
        pygame.draw.ellipse(surface, black, (cx - 5, cy + 5, 8, 6))
        pygame.draw.line(surface, black, (cx + 2, cy + 8), (cx + 2, cy - 12), 2)
        pygame.draw.arc(surface, black, (cx, cy - 15, 10, 10), 3.5, 6, 2)
    else:
        pygame.draw.ellipse(surface, black, (cx - 12, cy + 5, 8, 6))
        pygame.draw.ellipse(surface, black, (cx + 2, cy + 5, 8, 6))
        pygame.draw.line(surface, black, (cx - 5, cy + 8), (cx - 5, cy - 10), 2)
        pygame.draw.line(surface, black, (cx + 9, cy + 8), (cx + 9, cy - 10), 2)
        pygame.draw.line(surface, black, (cx - 5, cy - 10), (cx + 9, cy - 10), 2)


def draw_text(surface, text, cx, cy, font_size=20):
    """Draws text on the surface."""
    black = (0, 0, 0)
    try:
        font = pygame.font.SysFont(None, font_size)
        text_surf = font.render(text, True, black)
        text_rect = text_surf.get_rect(center=(cx, cy))
        surface.blit(text_surf, text_rect)
    except Exception:
        # Fallback if font fails: draw a simple box/line to represent text
        pygame.draw.rect(surface, black, (cx - 15, cy - 5, 30, 10), 1)


def generate_data(output_dir="data", num_samples_per_class=300):
    """Generates images for dog, flower, and background classes."""
    pygame.init()
    pygame.font.init()
    img_size = 64

    # Define backgrounds
    backgrounds = [
        ("white", (255, 255, 255), None),
        ("red", (255, 255, 255), (255, 100, 100)),  # white base, red crosshatch
        ("green", (255, 255, 255), (100, 200, 100)),  # white base, green crosshatch
    ]

    # Define classes
    # Background class will be handled specially
    other_classes = [("dog", draw_dog), ("flower", draw_flower)]

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Generate dog and flower images
    for class_name, draw_func in other_classes:
        class_dir = output_path / class_name
        class_dir.mkdir(exist_ok=True)
        print(f"Generating {num_samples_per_class} images for class: {class_name}")

        for i in range(num_samples_per_class):
            bg_name, bg_color, hatch_color = backgrounds[i % len(backgrounds)]

            surface = pygame.Surface((img_size, img_size))
            surface.fill(bg_color)

            if hatch_color:
                draw_crosshatch(surface, hatch_color)

            # Randomly add notes or text to some dog/flower images (to match environment variety)
            if np.random.random() < 0.2:
                draw_note(surface, 50, 15, single=np.random.random() > 0.5)
            if np.random.random() < 0.1:
                draw_text(surface, "Start", 32, 32)

            # Add some slight variation in position
            offset_x = np.random.randint(-2, 3)
            offset_y = np.random.randint(-2, 3)

            draw_func(surface, img_size // 2 + offset_x, img_size // 2 + offset_y)

            img_path = class_dir / f"{class_name}_{bg_name}_{i}.png"
            pygame.image.save(surface, str(img_path))

    # Generate background images
    class_name = "background"
    class_dir = output_path / class_name
    class_dir.mkdir(exist_ok=True)
    print(f"Generating {num_samples_per_class} images for class: {class_name}")

    for i in range(num_samples_per_class):
        bg_name, bg_color, hatch_color = backgrounds[i % len(backgrounds)]

        surface = pygame.Surface((img_size, img_size))
        surface.fill(bg_color)

        if hatch_color:
            draw_crosshatch(surface, hatch_color)

        # Background can be empty, or have notes, or have text
        rand_val = np.random.random()
        if rand_val < 0.3:
            # Just background, do nothing more
            pass
        elif rand_val < 0.6:
            # Background with notes
            draw_note(surface, 50, 15, single=np.random.random() > 0.5)
            if np.random.random() < 0.3:  # sometimes two notes
                draw_note(surface, 50, 40, single=np.random.random() > 0.5)
        else:
            # Background with text
            text = "Start" if np.random.random() > 0.5 else "Goal"
            draw_text(surface, text, 32, 32)

        img_path = class_dir / f"{class_name}_{bg_name}_{i}.png"
        pygame.image.save(surface, str(img_path))

    pygame.quit()
    print(f"Data generation complete. Images saved to {output_dir}")


if __name__ == "__main__":
    generate_data(num_samples_per_class=300)
