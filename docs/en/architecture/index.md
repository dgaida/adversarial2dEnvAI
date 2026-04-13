# Architecture

This document describes the internal architecture of the CustomGrid environment and the complete task fulfillment workflow.

## System Overview

The environment is designed modularly to allow easy expansion of sensors and agents.

```mermaid
graph TD
    subgraph "CustomGrid System"
        Interface[AgentInterface]
        Env[CustomGridEnv]
        PF[ParticleFilter]
        Vision[VisionSensor]
        Renderer[PygameRenderer]
    end

    UserAgent[User Agent] --> Interface
    Interface --> Env
    Interface --> PF
    Interface --> Vision
    Env --> Renderer
    Vision -.-> PF
    PF -.-> Interface
```

## Complete Task Fulfillment Workflow

The following diagram shows the complete workflow of how various modules work together to translate a user task (e.g., via speech) into agent actions. This includes planned future modules like **Speech2Text** and the **Task Planner**.

```mermaid
graph TD
    %% Input Modules
    User([User]) -- "Voice Command / Text" --> Input{Input Interface}

    subgraph "Task Understanding & Planning"
        S2T[Speech2Text Module *] -- "Transcript" --> NLP[NLP / LLM Parser *]
        NLP -- "Target List" --> Planner[Task Planner / TSP Solver *]
    end

    Input -- "Audio" --> S2T
    Input -- "Text" --> NLP

    subgraph "Agent Control (AgentInterface)"
        AI[AgentInterface]
        PF[Particle Filter]
        Vision[Vision Sensor]
        Audio[Audio Sensor *]
    end

    Planner -- "Next Sub-goal" --> Agent[AI Agent / Controller]

    Agent -- "Action" --> AI
    AI -- "Raw Data" --> Vision
    AI -- "Raw Data" --> Audio
    AI -- "Color Measurement" --> PF

    Vision -- "Classification" --> PF
    Audio -- "Audio ID" --> PF

    PF -- "Estimated Position" --> Agent
    Agent -- "State Feedback" --> Planner

    subgraph "Simulation Core"
        Env[CustomGridEnv]
        Renderer[Renderer]
    end

    AI -- "Execute Action" --> Env
    Env -- "Update" --> Renderer
    Renderer -- "Visualization" --> User

    classDef future fill:#f9f,stroke:#333,stroke-dasharray: 5 5;
    class S2T,NLP,Planner,Audio future;

    linkStyle default stroke:#333,stroke-width:2px;
```
*\* These modules are part of the extended architectural concept for students.*

## Data Flow (Step Level)

The data flow during a single simulation step (`step`):

```mermaid
sequenceDiagram
    participant A as Agent
    participant I as AgentInterface
    participant E as CustomGridEnv
    participant PF as Particle Filter
    participant V as Vision Sensor

    A->>I: step(action)
    I->>PF: predict(action)
    I->>E: step(action)
    E-->>I: obs, reward, info
    I->>V: predict(cell)
    V-->>I: cnn_probs
    I->>PF: update(measurements)
    PF->>PF: resample()
    I->>A: return obs, reward, info + estimated_pos
```

## Class Hierarchy

The agents follow a protocol-based design.

```mermaid
classDiagram
    class Agent {
        <<Protocol>>
        +get_action(observation)
    }
    class BaseAgent {
        +action_space
        +env
    }
    class MinimaxAgent
    class ExpectimaxAgent
    class ChaseGhostAgent
    class RandomGhostAgent

    BaseAgent ..|> Agent
    MinimaxAgent --|> BaseAgent
    ExpectimaxAgent --|> BaseAgent
    ChaseGhostAgent --|> BaseAgent
    RandomGhostAgent --|> BaseAgent
```
