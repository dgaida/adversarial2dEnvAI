import os
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output
from typing import Optional, Type
from .interface import AgentInterface
from .agents.base_agent import Agent
from .agents.random_player_agent import RandomPlayerAgent

# Set dummy video driver for headless environment (Colab)
os.environ["SDL_VIDEODRIVER"] = "dummy"


class ColabGUI:
    """A GUI for the CustomGrid environment that runs in Google Colab."""

    def __init__(
        self,
        agent_class: Type[Agent] = RandomPlayerAgent,
        slip_probability: float = 0.1,
    ):
        """Initializes the ColabGUI.

        Args:
            agent_class (Type[Agent]): The agent class to use for the agent.
            slip_probability (float): Probability of slipping.
        """
        self.interface = AgentInterface(
            render=True,
            render_mode="rgb_array",
            slip_probability=slip_probability,
            use_particle_filter=True,
        )
        self.agent = agent_class(self.interface.get_action_space())
        self.obs = self.interface.reset()

        # Widgets
        self.output = widgets.Output()
        self.next_button = widgets.Button(
            description="Next Step", button_style="primary"
        )
        self.reset_button = widgets.Button(
            description="Reset Episode", button_style="warning"
        )
        self.pf_toggle = widgets.Checkbox(value=True, description="Show Particles")
        self.sensor_dropdown = widgets.Dropdown(
            options=[
                ("Neural Net", "cnn"),
                ("Color Sensor", "color"),
                ("Both", "both"),
            ],
            value="both",
            description="PF Sensors:",
        )
        self.stats_label = widgets.Label(value="Steps: 0 | Total Reward: 0.0")

        # Layout
        self.controls = widgets.VBox(
            [
                widgets.HBox([self.next_button, self.reset_button]),
                widgets.HBox([self.pf_toggle, self.sensor_dropdown]),
                self.stats_label,
            ]
        )

        # Callbacks
        self.next_button.on_click(self._on_next_click)
        self.reset_button.on_click(self._on_reset_click)
        self.pf_toggle.observe(self._on_pf_toggle_change, names="value")
        self.sensor_dropdown.observe(self._on_sensor_change, names="value")

    def _on_next_click(self, b):
        if self.interface.is_terminated():
            return

        action = self.agent.get_action(self.obs)
        self.obs, reward, done, info = self.interface.step(action)
        self._update_display()

    def _on_reset_click(self, b):
        self.obs = self.interface.reset()
        self._update_display()

    def _on_pf_toggle_change(self, change):
        self.interface.show_particles = change["new"]
        self._update_display()

    def _on_sensor_change(self, change):
        self.interface.pf_sensor_mode = change["new"]

    def _update_display(self):
        with self.output:
            clear_output(wait=True)
            img = self.interface.env.render()
            if img is not None:
                plt.figure(figsize=(10, 8))
                plt.imshow(img)
                plt.axis("off")
                plt.show()

            stats = self.interface.get_episode_stats()
            self.stats_label.value = (
                f"Steps: {stats['steps']} | Total Reward: {stats['total_reward']:.1f}"
            )
            if stats["terminated"]:
                if stats["reached_goal"]:
                    print("SUCCESS: Goal reached!")
                elif stats["caught_by_ghost"]:
                    print("FAILURE: Caught by ghost!")

    def run(self):
        """Displays the GUI and performs initial render."""
        display(self.controls, self.output)
        self._update_display()
