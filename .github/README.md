# Docker Images Release Process

## Container Images

The repo produces container images, each with its own Dockerfile under `containers/`. Each container is described by a `ci.json` file in its directory that drives CI behaviour — no workflow changes are needed when adding a new container.

| Image | Base | Purpose |
|---|---|---|
| **python-base** | `python:3.13-slim` | Foundation layer — system deps, user/group setup |
| **python-apps-base** | `python-base` | App runtime — installs the Arduino App Bricks `.whl` |

Other containers are specialized inference runners needed to support AI platforms or use-cases.

## Release Workflow

A single workflow (`docker-publish.yml`) is triggered by any `release/X.Y.Z` tag (or manually via `workflow_dispatch`). It always builds **all** containers.

### Tagging

Every released image receives a `:<version>` tag. A `:latest` tag is also applied unless the version contains `rc`, `alpha`, or `beta`, which means it's a prerelease.

### Compose file versioning

Bricks may contain compose files that instruct docker about their required infrastructure. In these compose files, `__BRICKS_RELEASE_VERSION__` placeholders may be present. These are substituted with the release version at wheel build time by `arduino-bricks-release`. The compose files bundled inside the wheel always reference the containers tagged during the same release process. If older or unrelated versions are needed, don't use the placeholder and use a specific version.

### Caching

Buildx registry cache is stored as `<image>:buildcache` (mode=max). This tag is always tagged and is therefore never removed by the weekly cleanup job.

## Downstream Cascade

Containers may form a two-level hierarchy (parent -> child).
Base containers list their dependents in the `downstream` ci.json field. The `detect` job uses this to split containers into two groups that build in the correct order:

- **Base containers** (`build` job): containers with no upstream being built in the same run.
- **Downstream containers** (`build-downstream` job): containers whose upstream is also in the same run; these wait for `build` to finish and receive `BASE_IMAGE_VERSION=<version>` so they use the freshly built upstream image.

## Adding a New Container

1. Create `containers/my-container/Dockerfile`
2. Create `containers/my-container/ci.json`:

```json
{
  "build_whl": false,
  "build_args": {},
  "downstream": []
}
```

3. Push a `release/X.Y.Z` tag — the workflow picks it up automatically.

To declare that another container depends on yours, add it to `downstream`:

```json
"downstream": ["my-other-container"]
```

> **Note**: any container listed in `downstream` must declare `ARG BASE_IMAGE_VERSION` and `ARG REGISTRY` in its Dockerfile. The CI passes the upstream image's tag via `BASE_IMAGE_VERSION` so the downstream image pulls the freshly built version, not `:latest`.

No workflow file changes are required.

## ci.json Reference

| Field | Type | Description |
|---|---|---|
| `build_whl` | bool | Download the `.whl` artifact before the Docker build (set `true` on containers that install it) |
| `build_args` | object | Additional Docker build args passed to the Dockerfile (key/value pairs) |
| `downstream` | string[] | Containers that depend on this one — they build after this container in the same run |

## Dev Build Workflow

`docker-build.yml` is triggered manually via `workflow_dispatch` with:

- `branch` — branch to build (defaults to the selected branch)
- `containers` — comma-separated container names to build, or `all`
- `tag` — optional custom image tag (defaults to `dev-<sanitized-branch-name>`)
- `skip_cache` — rebuild without docker registry layer cache

Images are tagged `dev-<sanitized-branch>` and `dev-<sanitized-branch>-<run_number>`.

The same base / downstream split logic applies: base containers build first in parallel, downstream containers wait and receive `BASE_IMAGE_VERSION=<branch-tag>` as a build arg.

## Image Cleanup

`docker-cleanup.yml` runs two independent jobs:

| Trigger | Job | What it does |
|---|---|---|
| Branch deletion | `cleanup` | Deletes all GHCR versions tagged with `dev-<deleted-branch>` (including run-number suffixed tags and build caches) |
| Weekly (Sunday 03:00 UTC) / manual | `prune-untagged` | Deletes untagged container versions (orphaned buildx cache blobs). Any version carrying any tag is preserved. Each candidate is re-verified as still untagged before deletion. |

The manual `workflow_dispatch` for the weekly job accepts a `dry_run` flag to preview what would be deleted without removing anything.

## Build Characteristics

- **Single platform**: All images target `linux/arm64` only
- **Registry**: `ghcr.io/arduino/app-bricks/`
- **Caching**: Buildx registry cache (`type=registry`, `mode=max`)
- **Release assets**: The `.whl` is uploaded to the GitHub Release

## Image Size Monitoring

`calculate-size-delta.yml` is a manual workflow that builds both `python-base` and `python-apps-base`, measures their sizes using a local Docker registry, and posts a comment on the associated PR. If no PR is found, it falls back to the GitHub Actions Job Summary.
