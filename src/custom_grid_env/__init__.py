from .env import CustomGridEnv
from .interface import AgentInterface
from .agents.base_agent import Agent
from .agents.chase_ghost_agent import ChaseGhostAgent
from .agents.random_ghost_agent import RandomGhostAgent
from .agents.random_player_agent import RandomPlayerAgent
import gymnasium as gym

__all__ = [
    "CustomGridEnv",
    "AgentInterface",
    "Agent",
    "ChaseGhostAgent",
    "RandomGhostAgent",
    "RandomPlayerAgent",
]

gym.envs.registration.register(
    id="CustomGrid-v0",
    entry_point="custom_grid_env.env:CustomGridEnv",
)
