import pygame
import gymnasium as gym
from typing import Optional, Tuple, Dict, Any, Type
from .env import CustomGridEnv
from .agents.base_agent import Agent
from .agents.chase_ghost_agent import ChaseGhostAgent
from .particle_filter import ParticleFilter
from .logger import get_logger

logger = get_logger(__name__)


class AgentInterface:
    """Interface for AI agents to interact with the CustomGridEnv.

    Usage:
        interface = AgentInterface(render=True, slip_probability=0.2)
        obs = interface.reset()
        while not interface.is_terminated():
            action = your_agent.get_action(obs)
            obs, reward, done, info = interface.step(action)
        results = interface.get_episode_stats()
        interface.close()

    Attributes:
        env (CustomGridEnv): The gymnasium environment.
        render_enabled (bool): Whether to render the environment.
        step_delay (int): Delay between steps in milliseconds.
        total_reward (float): Cumulative reward in the current episode.
        terminated (bool): Whether the episode has terminated.
        truncated (bool): Whether the episode was truncated.
        episode_steps (int): Number of steps taken in the current episode.
        last_info (dict): Information from the last step.
    """

    def __init__(
        self,
        render: bool = True,
        render_mode: Optional[str] = None,
        step_delay: int = 100,
        slip_probability: float = 0.2,
        ghost_agent_class: Optional[Type[Agent]] = None,
        use_particle_filter: bool = True,
        pf_num_particles: int = 200,
        pf_sensor_mode: str = "both",  # 'color', 'cnn', or 'both'
        show_particles: bool = True,
    ):
        """Initializes the AgentInterface.

        Args:
            render (bool): Whether to render the graphical display. Defaults to True.
            render_mode (str, optional): The mode to render with ("human" or "rgb_array").
                Defaults to "rgb_array" if render is True and no mode is provided.
            step_delay (int): Milliseconds to wait between steps when rendering. Defaults to 100.
            slip_probability (float): Probability of slipping. Defaults to 0.2.
            ghost_agent_class (type, optional): Class for ghost agent. Defaults to ChaseGhostAgent.
        """
        if render_mode is None:
            render_mode = "rgb_array" if render else None

        self.env = CustomGridEnv(
            render_mode=render_mode, slip_probability=slip_probability
        )
        self.render_enabled = render
        self.step_delay = step_delay
        self.total_reward = 0.0
        self.terminated = False
        self.truncated = False
        self.episode_steps = 0
        self.last_info = {}

        if ghost_agent_class is None:
            self._ghost_agent = ChaseGhostAgent(self.env.action_space)
        else:
            self._ghost_agent = ghost_agent_class(self.env.action_space)

        self.use_particle_filter = use_particle_filter
        self.pf_sensor_mode = pf_sensor_mode
        self.show_particles = show_particles
        self.pf = None
        if self.use_particle_filter:
            self.pf = ParticleFilter(
                rows=self.env.rows, cols=self.env.cols, num_particles=pf_num_particles
            )

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Resets the environment for a new episode.

        Args:
            seed (int, optional): Random seed for reproducibility.

        Returns:
            dict: Initial observation for the agent.
        """
        obs, info = self.env.reset(seed=seed)
        self.total_reward = 0.0
        self.terminated = False
        self.truncated = False
        self.episode_steps = 0
        self.last_info = info

        if self.pf:
            # Re-initialize PF with original particle count
            self.pf = ParticleFilter(
                rows=self.env.rows,
                cols=self.env.cols,
                num_particles=self.pf.num_particles,
            )
            # Initial update with reset measurement
            self._update_pf(info)

        if self.render_enabled:
            self._render_with_pf()

        return obs

    def _update_pf(self, info: Dict[str, Any]):
        """Updates the particle filter with the latest measurements."""
        if not self.pf:
            return

        # Trigger CNN prediction if missing but required and renderer is available
        cnn_probs = info.get("cnn_probs")
        if (
            cnn_probs is None
            and self.pf_sensor_mode in ["cnn", "both"]
            and self.env.renderer
        ):
            current_cell = self.env.grid[self.env.agent_pos[0], self.env.agent_pos[1]]
            prediction_info = self.env.renderer._get_cnn_prediction(current_cell)
            if prediction_info:
                cnn_probs = prediction_info["probs"]
                info["cnn_probs"] = cnn_probs
                info["cnn_prediction"] = prediction_info["prediction"]

        measurements = {
            "color_measurement": info.get("color_measurement"),
            "cnn_probs": cnn_probs,
        }
        cnn_class_names = (
            self.env.renderer.class_names
            if self.env.renderer
            else ["dog", "flower", "background"]
        )
        self.pf.update(
            measurements,
            self.pf_sensor_mode,
            self.env.grid,
            cnn_class_names,
        )
        self.pf.resample()

    def _render_with_pf(self):
        """Renders the environment including PF data if enabled."""
        if self.pf and self.show_particles:
            self.env.info["particles"] = self.pf.get_particles()
            self.env.info["show_particles"] = True
        self.env.render()

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Takes a step in the environment (agent moves, then ghost moves automatically).

        Args:
            action (int): Agent action (0=left, 1=down, 2=right, 3=up).

        Returns:
            tuple: (observation, reward, done, info)
        """
        if self.terminated or self.truncated:
            raise RuntimeError(
                "Episode has ended. Call reset() to start a new episode."
            )

        combined_info = {}
        total_step_reward = 0.0

        # Agent's turn
        logger.debug(f"Agent's turn. action={action}")

        if self.pf:
            self.pf.predict(
                action, self.env.slip_probability, self.env._is_move_valid
            )

        obs, reward, self.terminated, self.truncated, info = self.env.step(action)
        total_step_reward += reward

        if self.pf:
            self._update_pf(info)

        if self.render_enabled:
            logger.debug("Rendering after agent's turn.")
            self._render_with_pf()
            pygame.time.wait(self.step_delay)

        logger.debug(
            f"env.info after agent's turn and potential render: {self.env.info}"
        )
        combined_info.update(self.env.info)

        if self.terminated:
            self.total_reward += total_step_reward
            self.episode_steps += 1
            self.last_info = combined_info
            return obs, float(total_step_reward), True, combined_info

        # Ghost's turn
        logger.debug("Ghost's turn.")
        ghost_obs = self.env._get_ghost_obs()
        ghost_action = self._ghost_agent.get_action(ghost_obs)

        obs, reward, self.terminated, self.truncated, info = self.env.step(ghost_action)

        if self.render_enabled:
            logger.debug("Rendering after ghost's turn.")
            self._render_with_pf()
            pygame.time.wait(self.step_delay)

        logger.debug(
            f"env.info after ghost's turn and potential render: {self.env.info}"
        )
        combined_info.update(self.env.info)

        if info.get("caught_by_ghost"):
            total_step_reward += reward

        self.total_reward += total_step_reward
        self.episode_steps += 1
        self.last_info = combined_info

        return (
            obs,
            float(total_step_reward),
            self.terminated or self.truncated,
            combined_info,
        )

    def is_terminated(self) -> bool:
        """Checks if the current episode has ended.

        Returns:
            bool: True if terminated or truncated, False otherwise.
        """
        return self.terminated or self.truncated

    def get_episode_stats(self) -> Dict[str, Any]:
        """Gets statistics for the current/last episode.

        Returns:
            dict: Episode statistics.
        """
        return {
            "total_reward": self.total_reward,
            "steps": self.episode_steps,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "reached_goal": self.last_info.get("reached_goal", False),
            "caught_by_ghost": self.last_info.get("caught_by_ghost", False),
        }

    def get_action_space(self) -> gym.spaces.Space:
        """Gets the action space.

        Returns:
            gym.spaces.Space: The action space.
        """
        return self.env.action_space

    def get_observation_space(self) -> gym.spaces.Space:
        """Gets the observation space.

        Returns:
            gym.spaces.Space: The observation space.
        """
        return self.env.observation_space

    def get_reward_structure(self) -> Dict[str, Any]:
        """Gets the reward structure for the environment.

        Returns:
            dict: The reward structure.
        """
        return self.env.get_reward_structure()

    def close(self):
        """Cleans up resources."""
        self.env.close()
