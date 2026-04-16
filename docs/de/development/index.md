# Entwicklung

## Mitwirken

Beiträge sind willkommen! Bitte beachte folgende Standards:

- **Docstrings**: Google-Stil ist obligatorisch.  
- **Tests**: Neue Features müssen durch Tests in `tests/` abgedeckt werden.  
- **API Abdeckung**: Muss über 95% bleiben.  

## Release Prozess

1. Version in `pyproject.toml` erhöhen.  
2. Changelog via `git-cliff` generieren.  
3. Dokumentation via GitHub Actions deployen.  
