import numpy as np
from typing import List, Dict, Any


class ParticleFilter:
    """A particle filter for agent localization in the CustomGrid environment."""

    def __init__(self, rows: int, cols: int, num_particles: int = 200):
        """Initializes the ParticleFilter.

        Args:
            rows (int): Number of rows in the grid.
            cols (int): Number of columns in the grid.
            num_particles (int): Number of particles.
        """
        self.rows = rows
        self.cols = cols
        self.num_particles = num_particles
        self.particles = []
        self.weights = np.ones(num_particles) / num_particles

        # Initialize particles randomly across the grid
        for _ in range(num_particles):
            r = np.random.randint(0, rows)
            c = np.random.randint(0, cols)
            self.particles.append([r, c])

    def predict(
        self, action: int, slip_prob: float, slip_type: str, env_is_move_valid_fn
    ):
        """Moves particles based on the agent's action and slip probability.

        Args:
            action (int): The intended action.
            slip_prob (float): Probability of slipping.
            slip_type (str): Type of slipping ("perpendicular" or "longitudinal").
            env_is_move_valid_fn (callable): Function to check if a move is valid.
        """
        perpendicular = {
            0: [1, 3],
            1: [0, 2],
            2: [1, 3],
            3: [0, 2],
        }

        new_particles = []
        for r, c in self.particles:
            # Determine actual actions for this particle
            actual_actions = []
            if np.random.random() < slip_prob:
                if slip_type == "longitudinal":
                    if np.random.random() < 0.5:
                        # Stay in place (0 steps)
                        actual_actions = []
                    else:
                        # Move twice (2 steps)
                        actual_actions = [action, action]
                else:
                    # Perpendicular
                    actual_actions = [np.random.choice(perpendicular[action])]
            else:
                actual_actions = [action]

            curr_r, curr_c = r, c
            for act in actual_actions:
                new_r, new_c = curr_r, curr_c
                if act == 0:
                    new_c = curr_c - 1
                elif act == 1:
                    new_r = curr_r + 1
                elif act == 2:
                    new_c = curr_c + 1
                elif act == 3:
                    new_r = curr_r - 1

                if env_is_move_valid_fn([curr_r, curr_c], [new_r, new_c]):
                    curr_r, curr_c = new_r, new_c

            new_particles.append([curr_r, curr_c])

        self.particles = new_particles

    def update(
        self,
        measurements: Dict[str, Any],
        sensor_mode: str,
        grid: np.ndarray,
        cnn_class_names: List[str],
    ):
        """Updates particle weights based on measurements.

        Args:
            measurements (dict): Contains 'color_measurement' and 'cnn_probs'.
            sensor_mode (str): 'color', 'cnn', or 'both'.
            grid (np.ndarray): The environment grid.
            cnn_class_names (list): List of class names for the CNN.
        """
        color_meas = measurements.get("color_measurement")
        cnn_probs = measurements.get("cnn_probs")

        for i, (r, c) in enumerate(self.particles):
            prob = 1.0
            cell = grid[r, c]

            # Color sensor update
            if sensor_mode in ["color", "both"] and color_meas is not None:
                actual_color = cell["colour"]
                if color_meas == actual_color:
                    prob *= 0.8
                else:
                    prob *= 0.1

            # CNN update
            if sensor_mode in ["cnn", "both"] and cnn_probs is not None:
                # Map cell content to CNN class index
                if "dog" in cell["items"]:
                    true_class_idx = cnn_class_names.index("dog")
                elif "flower" in cell["items"]:
                    true_class_idx = cnn_class_names.index("flower")
                else:
                    true_class_idx = cnn_class_names.index("background")

                # Likelihood is the probability the CNN gives to the true class of THIS particle's cell
                # Wait, PF update is: p(measurement | state).
                # Here 'state' is the particle's position.
                # Measurement is the CNN output (probabilities).
                # If the particle is at (r, c), how likely are we to get the observed CNN output?
                # This is tricky because the CNN output is a distribution.
                # Simplified: we assume the CNN is a noisy classifier.
                # The 'measurement' we use is the predicted class from the CNN (the one with highest prob).
                # But the user asked for "predictions of the neural network as measurements".
                # Let's use the probability the CNN assigned to the class that ACTUALLY exists at (r,c).
                prob *= cnn_probs[true_class_idx]

            self.weights[i] *= prob

        # Normalize weights
        sum_weights = np.sum(self.weights)
        if sum_weights > 0:
            self.weights /= sum_weights
        else:
            self.weights = np.ones(self.num_particles) / self.num_particles

    def resample(self):
        """Resamples particles based on their weights."""
        indices = np.random.choice(
            range(self.num_particles), size=self.num_particles, p=self.weights
        )
        self.particles = [self.particles[i] for i in indices]
        self.weights = np.ones(self.num_particles) / self.num_particles

    def get_particles(self) -> List[List[int]]:
        """Returns the list of particles."""
        return self.particles
