# Docker Images Release Process

## Container Images

The repo produces container images, each with its own Dockerfile under `containers/`. All build
configuration lives in `docker-bake.hcl` at the repo root: build context, build args, platform, cache,
and the release groups. Both workflows build through `docker buildx bake`, so
`docker buildx bake <container>` reproduces a CI build locally and `docker buildx bake --print` shows
the resolved definition.

The dependency graph between containers is not declared anywhere: it is derived from each Dockerfile's
`FROM ${REGISTRY}app-bricks/<parent>:${BASE_IMAGE_VERSION}` line by `scripts/container_deps.py`, which
both workflows and the SBOM tooling use. The same derivation yields the base image that delta SBOMs are
computed against.

| Image | Base | Purpose |
|---|---|---|
| **python-base** | `python:3.13-slim` | Foundation layer — system deps, user/group setup, fonts |
| **python-apps-base** | `python-base` | App runtime — installs the Arduino App Bricks `.whl`, Streamlit config |
| **ei-models-runner** | Edge Impulse inference image | AI/ML model inference with OOTB models |

## Release Triggers (Tag-Based)

A single workflow (`docker-publish.yml`) handles all container releases. It is triggered by any
`prefix/X.Y.Z` tag. The prefix selects the matching `group` in `docker-bake.hcl`; the containers
deriving from the group's members are rebuilt with it, in a second wave.

| Tag pattern | Containers | Extra behaviour |
|---|---|---|
| `release/X.Y.Z` | `python-base`, `python-apps-base`, `models-downloader` | Builds and uploads `.whl` to GitHub Release (displayed as `X.Y.Z`) |
| `ai/X.Y.Z` | `aihub-models-runner`, `ei-models-runner`, `ei-qnn-models-runner`, `python-slim` + derived runners | Auto-creates a PR to update compose file references |

If the pushed tag prefix matches no group, the workflow exits cleanly with no build.

## Adding a New Container

1. Create `containers/my-container/Dockerfile`. To build on another container of this repository,
   start it from the parent image — this is also what puts it in the downstream build wave:

```dockerfile
ARG REGISTRY
ARG BASE_IMAGE_VERSION
FROM ${REGISTRY}app-bricks/my-parent:${BASE_IMAGE_VERSION}
```

2. Add a matching target to `docker-bake.hcl` and list it in the `default` group plus the release
   group of the tag prefix that should release it:

```hcl
target "my-container" {
  inherits   = ["_common"]   # ["_downstream"] when building on another container of this repo
  context    = "containers/my-container"
  cache-from = cache_from("my-container")
  cache-to   = cache_to("my-container")
  args       = { MY_BUILD_ARG = "value" }
}
```

3. Push a tag `my-prefix/X.Y.Z` — the workflow picks it up automatically.

No workflow file changes required.

## Skip-Rebuild Logic

Every release checks whether the container's source files actually changed since the previous tag of
the same prefix:

- **Changed** → full Docker build and push
- **Unchanged** → `crane copy` re-tags the existing image to the new version (instant, no rebuild)

The watched paths are the container's bake contexts (its build context plus named contexts such as
`models`). The `wheel` context is special-cased to the wheel's inputs (`src/`, `pyproject.toml`,
`Taskfile.dist.yml`), since the wheel itself is a build artifact and not tracked in git.

## Dev Build Workflow

`docker-build.yml` is triggered manually via `workflow_dispatch` with:

- `containers` — comma-separated list of containers to build, or `all`
- `tag` — optional custom image tag (defaults to the sanitized branch name, e.g. `feat/my-feature` →
  `dev-feat-my-feature`, plus a run-number suffix)
- `skip_cache` — rebuild without importing the build cache

The selection is widened so related containers stay consistent: parents of selected containers are
rebuilt first, and containers deriving from a selected one are rebuilt after it. The two build waves
(`build` and `build-downstream` jobs) come from `scripts/container_deps.py waves`; downstream builds
receive `BASE_IMAGE_VERSION=<tag>` so they use the freshly built upstream image. No container names are
hardcoded in the workflows.

## Build Characteristics

- **Single platform**: All images target `linux/arm64` only
- **Registry**: `ghcr.io/arduino/app-bricks/`
- **Caching**: dev builds import and export a registry cache at `<image>:<image-tag>-buildcache`
  (`mode=max`); release builds run without cache and rely on the skip-rebuild logic instead
- **Release assets**: The `release/*` workflow also uploads the `.whl` to the GitHub Release via
  `softprops/action-gh-release`

## Image Size Monitoring

`calculate-size-delta.yml` is a manual workflow that builds both `python-base` and `python-apps-base`,
measures their sizes using a local Docker registry, and posts a comment on the associated PR. If no PR
is found, it falls back to the GitHub Actions Job Summary.
