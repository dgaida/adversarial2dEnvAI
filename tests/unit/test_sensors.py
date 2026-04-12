import os
import numpy as np
from custom_grid_env.sensors import VisionSensor


def test_vision_sensor_init():
    sensor = VisionSensor()
    # Should at least have class names defined
    assert sensor.class_names == ["dog", "flower", "background"]
    # Model might or might not be loaded depending on environment,
    # but the logic should not crash.


def test_vision_sensor_predict_no_model():
    sensor = VisionSensor(model_path="non_existent_path.keras")
    assert sensor.model is None
    result = sensor.predict({"colour": 0, "items": []})
    assert result is None


def test_vision_sensor_drawing():
    sensor = VisionSensor()
    import pygame

    if not pygame.get_init():
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()

    surface = pygame.Surface((64, 64))
    cell = {"colour": 1, "items": ["dog"]}
    sensor._draw_cell_for_cnn(cell, surface)

    # Check that it's not all white (meaning something was drawn)
    arr = pygame.surfarray.array3d(surface)
    assert not np.all(arr == 255)
