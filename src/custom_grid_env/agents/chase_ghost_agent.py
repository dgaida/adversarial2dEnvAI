"""Ghost agent that chases the player."""

from .base_agent import BaseAgent


class ChaseGhostAgent(BaseAgent):
    """Ghost agent that chases the player using simple pathfinding.

    This replicates the original built-in ghost behavior.
    """

    def get_action(self, observation: dict) -> int:
        """Choose action to move toward the agent.

        Args:
            observation (dict): Ghost's observation with 'agent_relative_pos'.

        Returns:
            int: Action (0=left, 1=down, 2=right, 3=up).
        """
        agent_relative = observation["agent_relative_pos"]
        row_diff = agent_relative[0]  # Positive = agent is below ghost
        col_diff = agent_relative[1]  # Positive = agent is to the right of ghost

        neighbors = observation["neighbors"]

        # Prioritize direction with larger distance
        if abs(row_diff) >= abs(col_diff):
            # Try vertical first
            if row_diff > 0 and neighbors["down"]["accessible"]:
                return 1  # down
            elif row_diff < 0 and neighbors["up"]["accessible"]:
                return 3  # up
            # Fall back to horizontal
            if col_diff > 0 and neighbors["right"]["accessible"]:
                return 2  # right
            elif col_diff < 0 and neighbors["left"]["accessible"]:
                return 0  # left
        else:
            # Try horizontal first
            if col_diff > 0 and neighbors["right"]["accessible"]:
                return 2  # right
            elif col_diff < 0 and neighbors["left"]["accessible"]:
                return 0  # left
            # Fall back to vertical
            if row_diff > 0 and neighbors["down"]["accessible"]:
                return 1  # down
            elif row_diff < 0 and neighbors["up"]["accessible"]:
                return 3  # up

        # If no good move, try any accessible direction
        for action, direction in [(3, "up"), (2, "right"), (1, "down"), (0, "left")]:
            if neighbors[direction]["accessible"]:
                return action

        # No move possible, stay in place
        return 0
