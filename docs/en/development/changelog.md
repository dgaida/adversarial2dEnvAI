# Changelog Workflow

We use **Conventional Commits** and **git-cliff** to automatically generate our changelog.

## Commit Message Format

Messages should follow this format:
`<type>(<scope>): <description>`

### Types  
- `feat`: A new feature  
- `fix`: A bugfix  
- `docs`: Documentation changes  
- `style`: Formatting, missing semicolons, etc.  
- `refactor`: Code refactoring without functional changes  
- `test`: Adding or changing tests  
- `chore`: Updates to build processes or auxiliary tools  

## Generation
The changelog is automatically updated with each release.
