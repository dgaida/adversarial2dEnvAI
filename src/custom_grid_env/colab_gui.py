"""GUI for the CustomGrid environment in Google Colab."""

import os
import time
import re
import json
import threading
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output
from sklearn.neighbors import KernelDensity
from typing import Type, List, Tuple, Dict, Any
from .interface import AgentInterface
from .logger import get_logger

logger = get_logger(__name__)
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
        self.interface.render_enabled = False
        self.agent = agent_class(
            self.interface.get_action_space(), env=self.interface.env
        )
        self.obs = self.interface.reset()
        self.planner = TaskPlanner(self.interface.env)

        # Planning & Execution State
        self.planned_targets: List[Tuple[int, int]] = []
        self.visited_mask: List[bool] = []
        self.executing = False
        self.paused = False

        # Output Widgets
        self.output = widgets.Output()
        self.llm_output_area = widgets.Textarea(
            value="",
            placeholder="LLM Response will appear here...",
            description="LLM Answer:",
            disabled=True,
            layout=widgets.Layout(width="100%", height="100px"),
        )
        self.target_status_area = widgets.VBox(
            [widgets.Label(value="No plan yet.")],
            layout=widgets.Layout(border="1px solid gray", padding="5px", width="100%"),
        )

        # Action Buttons
        self.next_button = widgets.Button(
            description="Next Step", button_style="primary"
        )
        self.reset_button = widgets.Button(
            description="Reset Episode", button_style="warning"
        )
        self.plan_button = widgets.Button(description="Plan", button_style="info")
        self.execute_button = widgets.Button(
            description="Execute", button_style="success"
        )
        self.pause_button = widgets.Button(description="Pause", button_style="danger")

        # Configuration Widgets
        self.pf_toggle = widgets.Checkbox(value=True, description="Show Particles")
        self.deterministic_toggle = widgets.Checkbox(
            value=False, description="Deterministic (Movement only)"
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

        self.task_input = widgets.Textarea(
            value=(
                "Besuche in optimaler Reihenfolge die folgenden drei Felder und kehre zum Ausgangsort zurück:\n"
                "- das Feld in dem man Klaviermusik hört, \n"
                "- das Feld wo man das Bild des Hundes sieht und Rockmusik hört \n"
                "- das Feld mit dem Schriftzug 'Ziel'"
            ),
            placeholder="Task description",
            description="Task:",
            layout=widgets.Layout(width="100%", height="100px"),
        )

        # Layout Grouping
        ghost_box = widgets.VBox(
            [
                widgets.HTML("<b>Ghost Settings</b>"),
                self.use_ghost_toggle,
                self.ghost_dropdown,
            ],
            layout=widgets.Layout(border="1px solid lightgray", padding="5px"),
        )
        pf_uncertainty_box = widgets.VBox(
            [
                widgets.HTML("<b>Uncertainty & PF</b>"),
                self.deterministic_toggle,
                self.slip_type_dropdown,
                self.pf_toggle,
                self.sensor_dropdown,
                self.color_quality_dropdown,
            ],
            layout=widgets.Layout(border="1px solid lightgray", padding="5px"),
        )
        agent_box = widgets.VBox(
            [
                widgets.HTML("<b>Agent Settings</b>"),
                self.agent_dropdown,
                self.knowledge_dropdown,
            ],
            layout=widgets.Layout(border="1px solid lightgray", padding="5px"),
        )

        self.controls = widgets.VBox(
            [
                widgets.HBox([self.next_button, self.reset_button, self.stats_label]),
                widgets.HBox([agent_box, ghost_box, pf_uncertainty_box]),
                widgets.VBox(
                    [
                        widgets.HTML("<b>Task Planning</b>"),
                        self.task_input,
                        widgets.HBox(
                            [self.plan_button, self.execute_button, self.pause_button]
                        ),
                        self.llm_output_area,
                        widgets.HTML("<b>Planned Targets</b>"),
                        self.target_status_area,
                    ],
                    layout=widgets.Layout(border="1px solid lightgray", padding="5px"),
                ),
            ]
        )

        # Event handlers
        self.next_button.on_click(self._on_next_click)
        self.reset_button.on_click(self._on_reset_click)
        self.plan_button.on_click(self._on_plan_click)
        self.execute_button.on_click(self._on_execute_click)
        self.pause_button.on_click(self._on_pause_click)
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
        self._update_display()

    def _on_deterministic_change(self, change):
        """Callback for the deterministic toggle."""
        self.interface.env.deterministic = change["new"]
        self._update_display()

    def _on_use_ghost_change(self, change):
        """Callback for the use ghost toggle."""
        self.interface.env.use_ghost = change["new"]
        self._update_display()

    def _on_plan_click(self, b):
        """Callback for the 'Plan' button."""
        task = self.task_input.value
        self.llm_output_area.value = "Planning... please wait."

        # Identify targets using LLM
        targets, response = self.planner.identify_targets(task)
        self.llm_output_area.value = response

        if not targets:
            return

        # Solve TSP
        ordered_targets = self.planner.solve_tsp(
            tuple(self.interface.env.agent_pos), targets
        )

        # Full targets: ordered targets + return to start
        self.planned_targets = ordered_targets + [tuple(self.interface.env.start_pos)]
        self.visited_mask = [False] * len(self.planned_targets)

        self._update_target_status()
        self._update_display()

    def _update_target_status(self):
        """Updates the target status visualization."""
        labels = []
        for i, target in enumerate(self.planned_targets):
            status = "✅" if self.visited_mask[i] else "⭕"
            name = "Start" if i == len(self.planned_targets) - 1 else f"Target {i+1}"
            labels.append(widgets.Label(value=f"{status} {name}: {target}"))
        self.target_status_area.children = labels

    def _on_execute_click_threaded(self, b):
        """Callback for the 'Execute' button (threaded version)."""
        if not self.planned_targets:
            return

        if self.executing:
            return

        self.executing = True
        self.paused = False
        self.pause_button.description = "Pause"

        # Run execution loop in a separate thread
        threading.Thread(target=self._run_execution, daemon=True).start()

    def _on_execute_click(self, b):
        """Callback for the 'Execute' button."""
        if not self.planned_targets:
            return

        if self.executing:
            return

        self.executing = True
        self.paused = False
        self.pause_button.description = "Pause"

        # Synchronous execution to avoid Matplotlib threading issues
        # To keep UI responsive in Colab, we should use a thread, but the user
        # specifically requested synchronous mode due to matplotlib issues.
        self._run_execution()
    def _on_pause_click(self, b):
        """Callback for the 'Pause' button."""
        self.paused = not self.paused
        self.pause_button.description = "Resume" if self.paused else "Pause"

    def _run_execution(self):
        """Runs the execution loop."""
        try:
            for i, target in enumerate(self.planned_targets):
                if self.visited_mask[i]:
                    continue

                # Use current agent for execution
                self.interface.env.set_goal(target)
                self.interface.terminated = False
                self.interface.truncated = False

                # While the agent is not at the target, keep taking steps
                while True:
                    if not self.executing:
                        return

                    while self.paused:
                        time.sleep(0.1)
                        if not self.executing:
                            return

                    # Check if current position (estimated or actual) reached target
                    if self.knowledge_dropdown.value == "estimated" and self.interface.pf:
                        curr_pos = tuple(
                            self.interface.pf.get_estimated_position()["cell_pos"]
                        )
                    else:
                        curr_pos = tuple(self.interface.env.agent_pos)

                    logger.info(f"Target: {target}, Current Pos: {curr_pos}")
                    if curr_pos == target:
                        break

                    # Perform one step using current agent settings
                    self._on_next_click(None)
                    time.sleep(0.5)  # Delay for visualization in Colab

                    if self.interface.is_terminated():
                        stats = self.interface.get_episode_stats()
                        if stats.get("caught_by_ghost"):
                            logger.info("Execution loop interrupted: caught by ghost.")
                            return
                        # If goal reached but not the final target, keep going
                        break

                self.visited_mask[i] = True
                logger.info(f"Target {i+1} reached: {target}")
                self._update_target_status()
        finally:
            self.executing = False

    def _on_next_click(self, b):
        """Callback for the 'Next Step' button."""
        if self.interface.is_terminated():
            logger.info(
                "Execution loop interrupted: environment terminated (caught by ghost?)."
            )
        if self.interface.is_terminated():
            return

