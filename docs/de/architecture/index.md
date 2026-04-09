# Architektur

## Systemübersicht

```mermaid
graph TD
    User[Benutzer/Skript] --> Interface[AgentInterface]
    Interface --> Env[CustomGridEnv]
    Env --> Renderer[PygameRenderer]
```
