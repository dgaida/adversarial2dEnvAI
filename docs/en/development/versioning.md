# Versioning

This project uses `mike` to manage versioned documentation.

## Deploying a New Version

To publish a new version of the documentation:

```bash
mike deploy --push --update-aliases 1.0 latest
mike set-default --push latest
```

## Patching
Changes to older versions can be made by specifying the corresponding tag.
