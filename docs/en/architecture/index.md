# Architecture

## System Overview

```mermaid
graph TD
    User[User/Script] --> Interface[AgentInterface]
    Interface --> Env[CustomGridEnv]
    Env --> Renderer[PygameRenderer]
```
