# Changelog Workflow

Wir verwenden **Conventional Commits** und **git-cliff**, um unseren Changelog automatisch zu generieren.

## Commit-Nachrichten Format

Nachrichten sollten folgendem Format folgen:
`<typ>(<bereich>): <beschreibung>`

### Typen  
- `feat`: Ein neues Feature  
- `fix`: Ein Bugfix  
- `docs`: Änderungen an der Dokumentation  
- `style`: Formatierung, fehlende Semikolons, etc.  
- `refactor`: Code-Refactoring ohne funktionale Änderungen  
- `test`: Hinzufügen oder Ändern von Tests  
- `chore`: Updates an Build-Prozessen oder Hilfswerkzeugen  

## Generierung
Der Changelog wird bei jedem Release automatisch aktualisiert.
