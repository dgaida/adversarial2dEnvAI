# Changelog Workflow

Wir nutzen **Conventional Commits** und **git-cliff**, um unseren Changelog automatisch zu pflegen.

## Commit-Nachrichten Format

Nachrichten sollten dem Schema folgen:
`<type>(<scope>): <description>`

Beispiele:  
- `feat(agent): add Q-learning implementation`  
- `fix(gui): resolve rendering flicker`  
- `docs(tutorial): add value iteration guide`  

## Automatisierung

Bei jedem Release (Push eines Tags) generiert unser GitHub Action Workflow automatisch den `CHANGELOG.md`.
