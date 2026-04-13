import os
import numpy as np
import matplotlib.pyplot as plt
from custom_grid_env.interface import AgentInterface

# Set dummy driver for headless run
os.environ["SDL_VIDEODRIVER"] = "dummy"

def main():
    interface = AgentInterface(render=True, render_mode="rgb_array", use_particle_filter=True)
    obs = interface.reset()

    # Take a few steps to ensure data persistence and UI updates
    for i in range(3):
        action = 2  # Move right
        obs, reward, done, info = interface.step(action)

        # Capture the RGB array
        img = interface.env.render()
        if img is not None:
            plt.imsave(f"step_{i}.png", img)
            print(f"Saved step_{i}.png")

            # Print info to verify data preservation
            print(f"Step {i} info keys: {list(info.keys())}")
            if "estimated_pos" in info:
                print(f"Est Pos: {info['estimated_pos']['cell_pos']}")

    interface.close()

if __name__ == "__main__":
    main()
