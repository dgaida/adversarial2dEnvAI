# Versionierung

Dieses Projekt verwendet `mike` für die Verwaltung versionierter Dokumentation.

## Deployment einer neuen Version

Um eine neue Version der Dokumentation zu veröffentlichen:

```bash
mike deploy --push --update-aliases 1.0 latest
mike set-default --push latest
```

## Patching
Änderungen an alten Versionen können durch Angabe des entsprechenden Tags vorgenommen werden.
