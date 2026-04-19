"""GUI for the CustomGrid environment in Google Colab."""

import os
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output
from sklearn.neighbors import KernelDensity
from typing import Type
from .interface import AgentInterface
from .agents.base_agent import Agent
from .agents.random_player_agent import RandomPlayerAgent
from .agents.chase_ghost_agent import ChaseGhostAgent
from .agents.random_ghost_agent import RandomGhostAgent
from .agents.adversarial_agents import MinimaxAgent, ExpectimaxAgent, AdversarialAgent
from .agents.value_iteration_agent import ValueIterationAgent
from .planner import TaskPlanner

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
        self.agent = agent_class(
            self.interface.get_action_space(), env=self.interface.env
        )
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
        self.deterministic_toggle = widgets.Checkbox(
            value=False, description="Deterministic"
        )
        self.use_ghost_toggle = widgets.Checkbox(value=True, description="Use Ghost")

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
            value="longitudinal",
            description="Slip Type:",
        )
        self.agent_dropdown = widgets.Dropdown(
            options=[
                ("Minimax", "minimax"),
                ("Expectimax", "expectimax"),
                ("Value Iteration", "value_iteration"),
                ("Random", "random"),
            ],
            value=(
                "random"
                if agent_class == RandomPlayerAgent
                else (
                    "minimax"
                    if agent_class == MinimaxAgent
                    else (
                        "expectimax"
                        if agent_class == ExpectimaxAgent
                        else "value_iteration"
                    )
                )
            ),
            description="Agent Type:",
        )
        self.ghost_dropdown = widgets.Dropdown(
            options=[
                ("Chase Agent", "chase"),
                ("Random", "random"),
                ("Minimax", "minimax"),
            ],
            value="chase",
            description="Ghost Type:",
        )
        self.knowledge_dropdown = widgets.Dropdown(
            options=[
                ("Actual State", "actual"),
                ("Estimated State", "estimated"),
            ],
            value="actual",
            description="Agent Knowledge:",
        )
        self.color_quality_dropdown = widgets.Dropdown(
            options=[
                ("100%", 1.0),
                ("90%", 0.9),
                ("80%", 0.8),
                ("70%", 0.7),
            ],
            value=0.8,
            description="Color Acc:",
        )
        self.stats_label = widgets.Label(value="Steps: 0 | Total Reward: 0.0")

        self.task_input = widgets.Text(
            value=(
                "Besuche in optimaler Reihenfolge die folgenden drei Felder und kehre zum Ausgangsort zurück:"
                "- das Feld in dem man Klaviermusik hört, "
                "- das Feld wo man das Bild des Hundes sieht und Rockmusik hört "
                "- das Feld mit dem Schriftzug 'Ziel'"
            ),
            placeholder="Task description",
            description="Task:",
            layout=widgets.Layout(width="80%"),
        )
        self.plan_button = widgets.Button(
            description="Plan & Execute", button_style="success"
        )

        self.planner = TaskPlanner(self.interface.env)

        # Layout
        self.controls = widgets.VBox(
            [
                widgets.HBox([self.next_button, self.reset_button]),
                widgets.HBox(
                    [self.pf_toggle, self.deterministic_toggle, self.use_ghost_toggle]
                ),
                widgets.HBox([self.sensor_dropdown, self.slip_type_dropdown]),
                widgets.HBox(
                    [
                        self.agent_dropdown,
                        self.ghost_dropdown,
                        self.color_quality_dropdown,
                    ]
                ),
                widgets.HBox([self.knowledge_dropdown, self.stats_label]),
                widgets.HBox([self.task_input, self.plan_button]),
            ]
        )

        # Event handlers
        self.next_button.on_click(self._on_next_click)
        self.reset_button.on_click(self._on_reset_click)
        self.plan_button.on_click(self._on_plan_click)
        self.pf_toggle.observe(self._on_pf_toggle_change, names="value")
        self.deterministic_toggle.observe(self._on_deterministic_change, names="value")
        self.use_ghost_toggle.observe(self._on_use_ghost_change, names="value")
        self.sensor_dropdown.observe(self._on_sensor_change, names="value")
        self.slip_type_dropdown.observe(self._on_slip_type_change, names="value")
        self.agent_dropdown.observe(self._on_agent_change, names="value")
        self.ghost_dropdown.observe(self._on_ghost_change, names="value")
        self.color_quality_dropdown.observe(
            self._on_color_quality_change, names="value"
        )
        self.knowledge_dropdown.observe(self._on_knowledge_change, names="value")

    def _on_knowledge_change(self, change):
        """Callback for the agent knowledge dropdown."""
        pass

    def _on_deterministic_change(self, change):
        """Callback for the deterministic toggle."""
        self.interface.env.deterministic = change["new"]

    def _on_use_ghost_change(self, change):
        """Callback for the use ghost toggle."""
        self.interface.env.use_ghost = change["new"]
        self._update_display()

    def _on_plan_click(self, b):
        """Callback for the 'Plan & Execute' button."""
        with self.output:
            task = self.task_input.value
            print(f"Planning for task: {task}")
            targets = self.planner.identify_targets(task)
            if not targets:
                print("Could not identify targets.")
                return

            print(f"Identified targets: {targets}")
            ordered_targets = self.planner.solve_tsp(
                tuple(self.interface.env.agent_pos), targets
            )
            print(f"Optimal order: {ordered_targets}")

            # Execute task (full path including return to start)
            full_targets = ordered_targets + [tuple(self.interface.env.start_pos)]

            for target in full_targets:
                print(f"Moving to target: {target}")
                self.interface.env.set_goal(target)
                current_pos = tuple(self.interface.env.agent_pos)
                path = self.planner.get_path(current_pos, target)
                for action in path:
                    if self.interface.is_terminated():
                        break
                    self.obs, reward, done, info = self.interface.step(action)
                    self._update_display()
                if self.interface.is_terminated():
                    break

    def _on_next_click(self, b):
        """Callback for the 'Next Step' button."""
        if self.interface.is_terminated():
            return

        if self.knowledge_dropdown.value == "estimated" and self.interface.pf:
            est_pos = self.interface.pf.get_estimated_position()["cell_pos"]
            self.agent.perceived_agent_pos = est_pos
            # Ghost position is also relative to estimated agent position in observation
            # but for adversarial agents we need the absolute ghost position.
            # We assume the agent knows where the ghost is relative to its estimate.
            actual_agent_pos = self.interface.env.agent_pos
            actual_ghost_pos = self.interface.env.ghost_pos
            rel_ghost = [
                actual_ghost_pos[0] - actual_agent_pos[0],
                actual_ghost_pos[1] - actual_agent_pos[1],
            ]
            self.agent.perceived_ghost_pos = [
                est_pos[0] + rel_ghost[0],
                est_pos[1] + rel_ghost[1],
            ]
        else:
            self.agent.perceived_agent_pos = None
            self.agent.perceived_ghost_pos = None

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

    def _on_agent_change(self, change):
        """Callback for the agent behavior dropdown."""
        if change["new"] == "minimax":
            self.agent = MinimaxAgent(
                self.interface.get_action_space(), env=self.interface.env
            )
        elif change["new"] == "expectimax":
            self.agent = ExpectimaxAgent(
                self.interface.get_action_space(), env=self.interface.env
            )
        elif change["new"] == "value_iteration":
            self.agent = ValueIterationAgent(
                self.interface.get_action_space(), env=self.interface.env
            )
        else:
            self.agent = RandomPlayerAgent(
                self.interface.get_action_space(), env=self.interface.env
            )

    def _on_color_quality_change(self, change):
        """Callback for the color quality dropdown."""
        self.interface.env.color_sensor_quality = change["new"]

    def _on_ghost_change(self, change):
        """Callback for the ghost behavior dropdown."""
        if change["new"] == "chase":
            self.interface.set_ghost_agent(ChaseGhostAgent)
        elif change["new"] == "minimax":
            self.interface.set_ghost_agent(MinimaxAgent)
        else:
            self.interface.set_ghost_agent(RandomGhostAgent)

    def _update_display(self):
        """Updates the display with the latest environment state and plots."""
        with self.output:
            clear_output(wait=True)

            # Precompute minimax/expectimax values if applicable
            agent_values = None
            ghost_values = None

            if isinstance(self.agent, AdversarialAgent):
                agent_values = np.zeros(
                    (self.interface.env.rows, self.interface.env.cols)
                )
                for r in range(self.interface.env.rows):
                    for c in range(self.interface.env.cols):
                        agent_values[r, c] = self.agent.get_value(
                            [r, c], self.interface.env.ghost_pos
                        )

            ghost_agent = self.interface._ghost_agent
            if isinstance(ghost_agent, AdversarialAgent):
                ghost_values = np.zeros(
                    (self.interface.env.rows, self.interface.env.cols)
                )
                for r in range(self.interface.env.rows):
                    for c in range(self.interface.env.cols):
                        # Ghost values from ghost's perspective?
                        # The request says "value of the cell for the ghost".
                        # AdversarialAgent.get_value returns value from agent perspective by default.
                        # MinimaxAgent.get_value calls _max_value(agent_pos, ghost_pos, ...)
                        # If we want the ghost's value for that cell, we should vary ghost_pos.
                        ghost_values[r, c] = ghost_agent.get_value(
                            self.interface.env.agent_pos, [r, c]
                        )

            self.interface.env.info["agent_values"] = agent_values
            self.interface.env.info["ghost_values"] = ghost_values

            img = self.interface.env.render()

            if img is not None:
                # Create a figure with two subplots: Grid and Probability Distribution
                fig, (ax1, ax2) = plt.subplots(
                    1, 2, figsize=(15, 8), gridspec_kw={"width_ratios": [1.2, 1]}
                )

                # Plot 1: Environment Grid
                ax1.imshow(img)
                ax1.axis("off")
                ax1.set_title("Environment Grid")

                # Plot 2: Multimodal Probability Distribution (KDE)
                if self.interface.pf:
                    rows, cols = self.interface.env.rows, self.interface.env.cols
                    particles = np.array(self.interface.pf.get_particles())
                    weights = self.interface.pf.weights

                    # Fit KDE to particles
                    # Use a small bandwidth for localized distribution
                    kde = KernelDensity(kernel="gaussian", bandwidth=0.3).fit(
                        particles, sample_weight=weights
                    )

                    # Create a fine grid for KDE evaluation
                    res = 100
                    x_lin = np.linspace(0, rows, res)
                    y_lin = np.linspace(0, cols, res)
                    Y_grid, X_grid = np.meshgrid(y_lin, x_lin)
                    grid_coords = np.vstack([X_grid.ravel(), Y_grid.ravel()]).T

                    # Evaluate log-density
                    log_dens = kde.score_samples(grid_coords)
                    dens = np.exp(log_dens).reshape(X_grid.shape)

                    # Plotting the contour plot
                    im = ax2.contourf(Y_grid, X_grid, dens, levels=20, cmap="viridis")
                    # Add contour lines for better visualization of the "elevation"
                    ax2.contour(
                        Y_grid, X_grid, dens, levels=10, colors="white", alpha=0.3
                    )
                    ax2.set_aspect("equal")
                    ax2.set_xlim(0, cols)
                    ax2.set_ylim(
                        rows, 0
                    )  # Inverted for grid coordinates (row 0 at top)

                    # Set explicit ticks for rows and columns
                    ax2.set_xticks(np.arange(0, cols))
                    ax2.set_yticks(np.arange(0, rows))

                    ax2.set_title("Estimated Probability Distribution (KDE)")
                    ax2.set_xlabel("Column")
                    ax2.set_ylabel("Row")
                    ax2.grid(True, linestyle="--", alpha=0.5)
                    fig.colorbar(im, ax2, label="Probability Density")

                    # Draw estimated position as a small filled circle
                    est_pos = self.interface.pf.get_estimated_position()
                    float_pos = est_pos["float_pos"]  # [row, col]
                    ax2.scatter(
                        float_pos[1],
                        float_pos[0],
                        color="red",
                        marker="X",
                        s=150,
                        edgecolors="white",
                        linewidths=2,
                        label="Estimated Position",
                        zorder=5,
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
