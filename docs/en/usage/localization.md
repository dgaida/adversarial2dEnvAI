# Localization

The environment provides a **Particle Filter** to estimate the agent's position when its exact location is unknown or when navigating with noisy sensors.

## Particle Filter

A Particle Filter (PF) is a Monte Carlo algorithm for estimating the state of a system. In this environment, it is used for **self-localization** of the agent on the grid.

### Mechanics

1. **Initialization**: Particles are distributed randomly across the entire grid.
2. **Prediction**: When the agent moves, each particle is moved according to the intended action, taking into account the movement uncertainty (slipping).
3. **Update**: Each particle's weight is updated based on how well the hypothetical observations at that particle's position match the actual sensor measurements.
4. **Resampling**: Particles with low weights are replaced by copies of particles with high weights.

## Sensors

The localization relies on two main sensor types:

### Color Sensor
The color sensor measures the color of the ground directly beneath the agent.
- **Accuracy**: 80% (correct color).
- **Noise**: 20% (distributed equally among the other two colors).
- **Colors**: White (0), Red (1), Green (2).

### CNN Classifier
The CNN classifier runs at every step, regardless of whether an item is present. If no item is present, it predicts the "background" class.
- **Classes**: Dog, Flower, Background.
- **Usage**: The probability assigned by the CNN to the class actually present in a particle's cell is used as the likelihood for that particle.

## Sensor Fusion

When using both sensors (`sensor_mode='both'`), the filter combines the measurements by assuming they are conditionally independent given the state. The joint likelihood is the product of the individual likelihoods:

$$p(z_{\text{color}}, z_{\text{cnn}} | s) = p(z_{\text{color}} | s) \cdot p(z_{\text{cnn}} | s)$$

## Assumptions

The Particle Filter makes the following assumptions about the environment:

### Map Assumption
The filter has **perfect knowledge of the map**. It knows:
- The dimensions of the grid.
- The exact location of all walls.
- The ground color of every cell.
- The locations of all items (Dogs and Flowers).

### Movement Uncertainty
The filter assumes a **stochastic motion model** matching the environment's slip probability. There are two types of slipping:

- **Perpendicular Slipping**: The agent moves in a perpendicular direction with probability $P_{\text{slip}}$ (split equally between the two perpendicular directions).
- **Longitudinal Slipping**: The agent moves in the same direction but either stays in place (moves 0 steps) or moves twice (moves 2 steps), each with probability $P_{\text{slip}} / 2$.
- **Intended Movement**: The agent moves exactly one step in the intended direction with probability $1 - P_{\text{slip}}$.
- **Walls**: If a move would lead into a wall or out of bounds, the agent (and thus the particles) remains in the current cell.

### Measurement Uncertainty
The filter assumes the following **likelihood models**:
- **Color Sensor**: $p(z_{\text{color}} | s) = 0.8$ if the measured color $z$ matches the map color at state $s$, and $0.1$ otherwise.
- **CNN**: $p(z_{\text{cnn}} | s) = \text{CNN\_prob}(\text{class at } s)$. The filter assumes that the probability output by the CNN for the true class at a location is the likelihood of that location.
