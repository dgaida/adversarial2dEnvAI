# Changelog Workflow

We use **Conventional Commits** and **git-cliff** to automatically maintain our changelog.

## Commit Message Format

Messages should follow the scheme:
`<type>(<scope>): <description>`

Examples:
- `feat(agent): add Q-learning implementation`
- `fix(gui): resolve rendering flicker`
- `docs(tutorial): add value iteration guide`

## Automation

With every release (tag push), our GitHub Action workflow automatically generates the `CHANGELOG.md`.
