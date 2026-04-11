import os
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output
from typing import Type
from .interface import AgentInterface
from .agents.base_agent import Agent
from .agents.random_player_agent import RandomPlayerAgent
from .agents.chase_ghost_agent import ChaseGhostAgent
from .agents.random_ghost_agent import RandomGhostAgent

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
        self.slip_type_dropdown = widgets.Dropdown(
            options=[
                ("Perpendicular", "perpendicular"),
                ("Longitudinal", "longitudinal"),
            ],
            value="perpendicular",
            description="Slip Type:",
        )
        self.ghost_dropdown = widgets.Dropdown(
            options=[
                ("Chase Agent", "chase"),
                ("Random", "random"),
            ],
            value="chase",
            description="Ghost Type:",
        )
        self.stats_label = widgets.Label(value="Steps: 0 | Total Reward: 0.0")

        # Layout
        self.controls = widgets.VBox(
            [
                widgets.HBox([self.next_button, self.reset_button]),
                widgets.HBox([self.pf_toggle, self.sensor_dropdown]),
                widgets.HBox([self.slip_type_dropdown, self.ghost_dropdown]),
                self.stats_label,
            ]
        )

        # Callbacks
        self.next_button.on_click(self._on_next_click)
        self.reset_button.on_click(self._on_reset_click)
        self.pf_toggle.observe(self._on_pf_toggle_change, names="value")
        self.sensor_dropdown.observe(self._on_sensor_change, names="value")
        self.slip_type_dropdown.observe(self._on_slip_type_change, names="value")
        self.ghost_dropdown.observe(self._on_ghost_change, names="value")

    def _on_next_click(self, b):
        """Callback for the 'Next Step' button."""
        if self.interface.is_terminated():
            return

        action = self.agent.get_action(self.obs)
        self.obs, reward, done, info = self.interface.step(action)
        self._update_display()

    def _on_reset_click(self, b):
        """Callback for the 'Reset Episode' button."""
        self.obs = self.interface.reset()
        self._update_display()

    def _on_pf_toggle_change(self, change):
        """Callback for the particle filter toggle."""
        self.interface.show_particles = change["new"]
        self._update_display()

    def _on_sensor_change(self, change):
        """Callback for the sensor selection dropdown."""
        self.interface.pf_sensor_mode = change["new"]

    def _on_slip_type_change(self, change):
        """Callback for the slip type dropdown."""
        self.interface.env.slip_type = change["new"]

    def _on_ghost_change(self, change):
        """Callback for the ghost behavior dropdown."""
        if change["new"] == "chase":
            self.interface.set_ghost_agent(ChaseGhostAgent)
        else:
            self.interface.set_ghost_agent(RandomGhostAgent)

    def _update_display(self):
        """Updates the display with the latest environment state and plots."""
        with self.output:
            clear_output(wait=True)
            img = self.interface.env.render()

            if img is not None:
                # Create a figure with two subplots: Grid and Probability Distribution
                fig, (ax1, ax2) = plt.subplots(
                    2, 1, figsize=(10, 14), gridspec_kw={"height_ratios": [1.5, 1]}
                )

                # Plot 1: Environment Grid
                ax1.imshow(img)
                ax1.axis("off")
                ax1.set_title("Environment Grid")

                # Plot 2: Multimodal Probability Distribution
                if self.interface.pf:
                    rows, cols = self.interface.env.rows, self.interface.env.cols
                    # Create a 2D histogram of particles to represent the distribution
                    particles = np.array(self.interface.pf.get_particles())
                    weights = self.interface.pf.weights

                    # We want a smooth representation, so we use a 2D histogram
                    # then potentially smooth it or use it as a heatmap.
                    hist, xedges, yedges = np.histogram2d(
                        particles[:, 1],
                        particles[:, 0],
                        bins=[cols, rows],
                        range=[[0, cols], [0, rows]],
                        weights=weights,
                    )

                    # Plotting the contour plot (Höhendiagramm)
                    x = np.linspace(0.5, cols - 0.5, cols)
                    y = np.linspace(0.5, rows - 0.5, rows)
                    X, Y = np.meshgrid(x, y)

                    # Ensure at least some levels exist even if hist is all zeros
                    max_val = np.max(hist)
                    levels = (
                        np.linspace(0, max_val, 20)
                        if max_val > 0
                        else np.linspace(0, 1, 20)
                    )

                    im = ax2.contourf(X, Y, hist.T, levels=levels, cmap="viridis")
                    ax2.set_aspect("equal")
                    ax2.set_xlim(0, cols)
                    ax2.set_ylim(rows, 0)  # Inverted for grid coordinates (row 0 at top)

                    ax2.set_title(
                        "Estimated Probability Distribution (Contour Plot)"
                    )
                    ax2.set_xlabel("Column")
                    ax2.set_ylabel("Row")
                    fig.colorbar(im, ax2, label="Probability Density")

                    # Draw estimated position as a small filled circle
                    est_pos = self.interface.pf.get_estimated_position()
                    float_pos = est_pos["float_pos"]  # [row, col]
                    ax2.scatter(
                        float_pos[1],
                        float_pos[0],
                        color="red",
                        s=100,
                        edgecolors="white",
                        label="Estimated Position",
                    )
                    ax2.legend()

                plt.tight_layout()
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
