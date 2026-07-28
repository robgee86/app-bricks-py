# Docker Images Release Process

## Container Images

The repo produces container images, each with its own Dockerfile under `containers/`. All build
configuration lives in `docker-bake.hcl` at the repo root: build context, build args, platform, cache,
and the release groups. Both workflows build through `docker buildx bake`, so
`docker buildx bake <container>` reproduces a CI build locally and `docker buildx bake --print` shows
the resolved definition.

The dependency graph between containers is declared in each Dockerfile's
`FROM ${REGISTRY}app-bricks/<parent>:${BASE_IMAGE_VERSION}` line, mirrored by a `parent_context()`
link in the bake target. Bake builds parents in-graph in dependency order, however deep the chain,
building shared parents once. `scripts/container_deps.py` derives the same graph from the Dockerfiles,
to widen the build selection and to resolve the base image delta SBOMs are computed against.
`task containers:tree` prints the hierarchy.

| Image | Base | Purpose |
|---|---|---|
| **python-base** | `python:3.13-slim` | Foundation layer — system deps, user/group setup, fonts |
| **python-apps-base** | `python-base` | App runtime — installs the Arduino App Bricks `.whl`, Streamlit config |
| **ei-models-runner** | Edge Impulse inference image | AI/ML model inference with OOTB models |

## Release Triggers (Tag-Based)

A single workflow (`docker-publish.yml`) handles all container releases. It is triggered by any
`prefix/X.Y.Z` tag. The prefix selects the matching `group` in `docker-bake.hcl`, widened with the
containers deriving from the group's members; one bake invocation rebuilds them all, ordered through
the parent links, reusing unchanged layers from the `release-buildcache` registry cache. A release
opens a single draft PR updating all compose file references.

| Tag pattern | Containers | Extra behaviour |
|---|---|---|
| `release/X.Y.Z` | `python-base`, `python-apps-base`, `models-downloader` | Builds and uploads `.whl` to GitHub Release (displayed as `X.Y.Z`) |
| `ai/X.Y.Z` | `aihub-models-runner`, `ei-models-runner`, `ei-qnn-models-runner`, `python-slim` + derived runners | Auto-creates a PR to update compose file references |

If the pushed tag prefix matches no group, the workflow exits cleanly with no build.

## Adding a New Container

1. Create `containers/my-container/Dockerfile`. To build on another container of this repository,
   start it from the parent image:

```dockerfile
ARG REGISTRY
ARG BASE_IMAGE_VERSION
FROM ${REGISTRY}app-bricks/my-parent:${BASE_IMAGE_VERSION}
```

2. Add a matching target to `docker-bake.hcl` and list it in the `default` group plus the release
   group of the tag prefix that should release it:

```hcl
target "my-container" {
  inherits   = ["_downstream"]   # ["_common"] when not building on a container of this repo
  context    = "containers/my-container"
  tags       = image_tags("my-container")
  cache-from = cache_from("my-container")
  cache-to   = cache_to("my-container")
  contexts   = parent_context("my-parent")   # only when building on a container of this repo
  args       = { MY_BUILD_ARG = "value" }
}
```

3. Push a tag `my-prefix/X.Y.Z` — the workflow picks it up automatically.

No workflow file changes required.

## Dev Build Workflow

`docker-build.yml` is triggered manually via `workflow_dispatch` with:

- `containers` — comma-separated list of containers to build, or `all`
- `tag` — optional custom image tag (defaults to the sanitized branch name, e.g. `feat/my-feature` →
  `dev-feat-my-feature`, plus a run-number suffix)
- `skip_cache` — rebuild without importing the build cache

The selection is widened so related containers stay consistent: parents of selected containers are
built and pushed too, and containers deriving from a selected one are rebuilt with it. A single bake
invocation builds everything — the parent links in `docker-bake.hcl` give bake the dependency order,
at any depth, with shared parents built once. No container names are hardcoded in the workflows.

## Build Characteristics

- **Single platform**: All images target `linux/arm64` only
- **Registry**: `ghcr.io/arduino/app-bricks/`
- **Caching**: builds import and export a registry cache (`mode=max`) — dev builds at
  `<image>:<image-tag>-buildcache` (per branch), release builds at `<image>:release-buildcache`, so
  releases only pay for the layers that actually changed since the previous release
- **Release assets**: The `release/*` workflow also uploads the `.whl` to the GitHub Release via
  `softprops/action-gh-release`

## Image Size Monitoring

`calculate-size-delta.yml` is a manual workflow that builds both `python-base` and `python-apps-base`,
measures their sizes using a local Docker registry, and posts a comment on the associated PR. If no PR
is found, it falls back to the GitHub Actions Job Summary.
