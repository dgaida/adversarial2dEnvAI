# Tutorial: Particle Filter and Sensor Fusion

This tutorial explains how the particle filter is used in the `CustomGrid` environment for agent localization. It is aimed at computer science students at the bachelor level.

## The Problem: Where am I?

In a realistic robotics application, the robot often does not know its exact position. It knows which action it has performed (e.g., "take one step forward"), but due to uncertainties (slipping on the floor), the result is stochastic.

The **particle filter** is an algorithm that uses a set of hypotheses (particles) to represent the likely position of the agent.

## How the Particle Filter Works

A particle filter works in a cycle of three steps:

1.  **Prediction**: Each particle is moved according to the agent's action. The motion model (including slip probability) is simulated.  
2.  **Correction (Update)**: Based on the sensor measurements, each particle is evaluated. Particles whose position matches the measurements well receive a higher weight.  
3.  **Resampling**: Particles with low weight are removed, while particles with high weight are multiplied. This way, the "cloud" concentrates on the most likely locations.  

## Sensor Fusion in CustomGrid

The particle filter in this environment combines two different sensor types (**sensor fusion**) to estimate the position:

### 1. The Color Sensor
The agent has a sensor that measures the ground color (white, red, green). This sensor has an accuracy of **80%**.  
- If a particle is on a red cell and the sensor reports "red", the probability for this particle increases.  
- If the sensor reports "green" even though the particle is on a red cell, the probability decreases.  

### 2. The CNN (Visual Recognition)
The trained Convolutional Neural Network provides probabilities for the classes `dog`, `flower`, and `background`.
The particle filter uses these predictions as measurements:  
- Each particle "looks" at the map: Which object is at my (hypothetical) position?  
- The likelihood of a particle is calculated from the probability that the CNN outputted for exactly this object.  

## Mathematical Combination

We assume that the sensors are conditionally independent. The total probability $P$ for a particle results from the product of the individual probabilities:

$$p(z_{\text{color}}, z_{\text{cnn}} | s) = p(z_{\text{color}} | s) \cdot p(z_{\text{cnn}} | s)$$

Through this combination, the agent can determine its position even if a single sensor is very noisy. For example, if the CNN is uncertain, the color sensor can often help to narrow down the position on the grid.

## Exercise for Students

1.  **Influence of Sensors**: Test the particle filter in the `Colab_GUI_Demo` with only the color sensor, only the CNN, and with both. Observe how quickly the particle cloud converges.  
2.  **Slip Models**: Compare "perpendicular" slipping with "longitudinal" slipping. Which model makes localization more difficult?  
3.  **Number of Particles**: Reduce the number of particles in the `AgentInterface`. At what number does the estimate become unstable?  
