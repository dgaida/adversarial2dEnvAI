import pygame
from typing import Optional, Tuple, Dict, Any, Type
from .env import CustomGridEnv
from .agents.base_agent import Agent
from .agents.chase_ghost_agent import ChaseGhostAgent


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

    def __init__(self, render: bool = True, step_delay: int = 100, slip_probability: float = 0.2, ghost_agent_class: Optional[Type[Agent]] = None):
        """Initializes the AgentInterface.

        Args:
            render (bool): Whether to render the graphical display. Defaults to True.
            step_delay (int): Milliseconds to wait between steps when rendering. Defaults to 100.
            slip_probability (float): Probability of slipping. Defaults to 0.2.
            ghost_agent_class (type, optional): Class for ghost agent. Defaults to ChaseGhostAgent.
        """
        self.env = CustomGridEnv(slip_probability=slip_probability)
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

        if self.render_enabled:
            self.env.render()

        return obs

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Takes a step in the environment (agent moves, then ghost moves automatically).

        Args:
            action (int): Agent action (0=left, 1=down, 2=right, 3=up).

        Returns:
            tuple: (observation, reward, done, info)
        """
        if self.terminated or self.truncated:
            raise RuntimeError("Episode has ended. Call reset() to start a new episode.")

        combined_info = {}
        total_step_reward = 0.0

        # Agent's turn
        obs, reward, self.terminated, self.truncated, info = self.env.step(action)
        total_step_reward += reward
        combined_info.update(info)

        if self.render_enabled:
            self.env.render()
            pygame.time.wait(self.step_delay)

        if self.terminated:
            self.total_reward += total_step_reward
            self.episode_steps += 1
            self.last_info = combined_info
            return obs, float(total_step_reward), True, combined_info

        # Ghost's turn
        ghost_obs = self.env._get_ghost_obs()
        ghost_action = self._ghost_agent.get_action(ghost_obs)

        obs, reward, self.terminated, self.truncated, info = self.env.step(ghost_action)
        combined_info.update(info)

        if self.render_enabled:
            self.env.render()
            pygame.time.wait(self.step_delay)

        if info.get('caught_by_ghost'):
            total_step_reward += reward

        self.total_reward += total_step_reward
        self.episode_steps += 1
        self.last_info = combined_info

        return obs, float(total_step_reward), self.terminated or self.truncated, combined_info

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

    def get_action_space(self):
        """Gets the action space.

        Returns:
            gym.spaces.Space: The action space.
        """
        return self.env.action_space

    def get_observation_space(self):
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
