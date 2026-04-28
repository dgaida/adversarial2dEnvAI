# Architektur

Dieses Dokument beschreibt die interne Architektur der CustomGrid-Umgebung und den vollständigen Workflow zur Aufgabenerfüllung.

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

## Vollständiger Workflow zur Aufgabenerfüllung

Das folgende Diagramm zeigt den kompletten Workflow, wie verschiedene Module zusammenarbeiten, um eine Benutzeraufgabe (z. B. per Sprache) in Aktionen des Agenten umzusetzen.

```mermaid
graph TD
    %% Input Module
    User([Benutzer]) -- "Sprachbefehl / Text" --> Input{Eingabe-Schnittstelle}

    subgraph "Aufgabenverständnis & Planung"
        S2T[Speech2Text Modul *] -- "Transkript" --> NLP[NLP / LLM Parser]
        NLP -- "Gelistete Ziele" --> Planner[Task Planner / TSP Solver]
    end

    Input -- "Audio" --> S2T
    Input -- "Text" --> NLP

    subgraph "Agentensteuerung (AgentInterface)"
        AI[AgentInterface]
        PF[Partikelfilter]
        Vision[Vision Sensor]
        Audio[Audio Sensor *]
    end

    Planner -- "Nächstes Teilziel" --> Agent[KI Agent / Controller]

    Agent -- "Aktion" --> AI
    AI -- "Rohdaten" --> Vision
    AI -- "Rohdaten" --> Audio
    AI -- "Farbmessung" --> PF

    Vision -- "Klassifizierung" --> PF
    Audio -- "Audio-ID" --> PF

    PF -- "Geschätzte Position" --> Agent
    Agent -- "Zustands-Feedback" --> Planner

    subgraph "Simulations-Kern"
        Env[CustomGridEnv]
        Renderer[Renderer]
    end

    AI -- "Aktion ausführen" --> Env
    Env -- "Update" --> Renderer
    Renderer -- "Visualisierung" --> User

    classDef future fill:#f9f,stroke:#333,stroke-dasharray: 5 5;
    class S2T,Audio future;

    linkStyle default stroke:#333,stroke-width:2px;
```
*\* Diese Module sind Teil des erweiterten Architektur-Konzepts für Studierende.*

## Datenfluss (Schritt-Ebene)

Der Datenfluss während eines einzelnen Simulationsschrittes (`step`):

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
    class Agent
    <<interface>> Agent
    Agent : +get_action(observation)

    class BaseAgent {
        +action_space
        +env
    }
    class MinimaxAgent
    class ExpectimaxAgent
    class ChaseGhostAgent
    class RandomGhostAgent
    class ValueIterationAgent
    class QLearningAgent

    BaseAgent ..|> Agent : implements
    MinimaxAgent --|> BaseAgent : inherits
    ExpectimaxAgent --|> BaseAgent : inherits
    ChaseGhostAgent --|> BaseAgent : inherits
    RandomGhostAgent --|> BaseAgent : inherits
    ValueIterationAgent --|> BaseAgent : inherits
    QLearningAgent --|> BaseAgent : inherits
```
