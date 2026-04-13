# Architektur

Dieses Dokument beschreibt die interne Architektur der CustomGrid-Umgebung.

## Systemübersicht

Die Umgebung ist modular aufgebaut, um eine einfache Erweiterung von Sensoren und Agenten zu ermöglichen.

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

## Datenfluss

Der Datenfluss während eines Schrittes (`step`) folgt einem strikten rundenbasierten Protokoll.

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

## Klassen-Hierarchie

Die Agenten folgen einem Protokoll-basierten Design.

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
