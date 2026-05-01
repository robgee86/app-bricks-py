# Arduino Apps Brick Library

Library is composed by configurable and reusable 'Bricks', based on optional infrastructure (executed via Docker Compose) and wrapping Python® code (to simplify code usage). 

## What is a Brick?

A **Brick** is a modular, reusable building block that provides specific functionality for Arduino applications. Each Brick is self-contained with standardized configuration, consistent APIs, and optional Docker service definitions.

## Directory Structure

Every Brick must follow this standardized directory structure:

```
src/arduino/app_bricks/brick_name/
└── [assets]                      # Optional: Static resources
├── examples/                     # Required: Usage examples
├── __init__.py                   # Required: Public API exports
├── brick_config.yaml             # Required: Brick metadata
├── brick_compose.yaml            # Optional: Docker infrastructure
├── brick_compose.<platform>.yaml # Optional: Docker infrastructure for a specific platform
│   ├── 1_basic_usage.py
│   ├── 2_advanced_usage.py
│   └── ...
├── [implementation_files.py]     # Brick logic
├── README.md                     # Required: Documentation
```

## Configuration variables

| Variable | Description |
|---|---|
| APP_HOME | Base application directory context |
| LOCAL_DEV | Switch logic for local library development |
| BRICKS_RELEASE_VERSION | Version stamped in the wheel and in compose file image tags (defaults to `0.0.0`) |

## Library development
To start the development, clone the repository and create a virtual environment.

Install the Taskfile CLI tool: https://taskfile.dev/installation/.

Run the following command to set up the development environment:

```sh
task init
```

This task will check the Python version and install development dependencies.

## Linting and formatting

You can use the Ruff CLI to safely auto-fix linting issues and format your code by running:

```sh
task lint
task fmt
```

Alternatively, to improve the development experience in VS Code, you can add a `.vscode` folder to the repository root containing the following JSON files:

`extensions.json`

```json
{
  "recommendations": [
    "charliermarsh.ruff",
    "github.vscode-pull-request-github",
    "ms-python.python",
    "tamasfe.even-better-toml"
  ],
  "unwantedRecommendations": [
    "ms-python.pylint"
  ]
}
```

`settings.json`

```json
{
    // Set the Python interpreter to the virtual environment
    "python.defaultInterpreterPath": "${workspaceFolder}/.venv",
    "flake8.enabled": false,  // Disable flake8 since we use ruff
    "ruff.enable": true,
    "python.testing.pytestArgs": [
        "tests"
    ],
    "python.testing.unittestEnabled": false,
    "python.testing.pytestEnabled": true,

    // Linting and fromatting settings on save
    "[python]": {
        // 1) use ruff as the default formatter
        "editor.defaultFormatter": "charliermarsh.ruff",
        
        // 2) automatically format the code on save
        // comment this setting if you don't want to automatically format your code on save
        "editor.formatOnSave": true,

        // 3) apply secure linter fixes on save
        // comment this setting if you don't want to automatically fix with the linter your code on save
        "editor.codeActionsOnSave": {
            "source.fixAll.ruff": "explicit",
        }
    }
}
```

After adding those files, VS Code will suggest installing the Python and Ruff extensions, which are properly configured for this project.

## Testing

All tests must be added in tests/ folder. To execute tests, run command:
```sh
task test
```

or, to execute specific tests, use:
```sh
task test:arduino/app_bricks
```

## Building the wheel file

Requires a working dev environment (see [Library development](#library-development) below).

To build the wheel:
```sh
task build
```

To stamp a specific version:
```sh
BRICKS_RELEASE_VERSION=1.2.3 task build
```

The default version is `0.0.0`. CI stamps the correct version by setting `BRICKS_RELEASE_VERSION` to the release tag value before running `task build`.

## Release

Release is based on tags pushed to `main`. A single workflow (`docker-publish.yml`) builds **all** containers when a `release/X.Y.Z` tag is pushed. The Python wheel file is uploaded to the GitHub Release.

**Prerelease**: if the version contains `rc`, `alpha`, or `beta`, images are tagged with the version number only — no `:latest` tag is pushed.

For dev builds, use the `docker-build.yml` manual workflow to build specific containers from a branch with a `dev-<branch>` tag.

See [`.github/README.md`](.github/README.md) for full CI documentation.

### Container layers

Library containers are built in layers: `python-base` provides system dependencies and is the foundation for `python-apps-base` (which installs the `.whl`). These layers are rebuilt together on every `release/X.Y.Z` tag but cached aggressively via buildx registry cache, so unchanged layers are reused across releases.

Base images are required to:
* reduce the amount of updated layers during a single library update
* promote reuse of existing layers in multiple builds
* cache pre-compiled python libraries as much as possible

Non-base images should start from common base images for performance and disk usage needs.

## License
This library is licensed under MPL-2.0.

See [LICENSE](./LICENSE.txt) file for the license text.

## SBOM (Software Bill of Materials)
Each release includes multiple SBOM files for each container:

- **base.spdx.json**: lists all installed packages, their versions, and licenses of the base image used for the released container. 
- **full.spdx.json**: lists all installed packages, their versions, and licenses of the released image.
- **delta.json**: lists packages added, modified and removed with respect to the base image when building the released image (i.e. difference between full and base images).

The base and full SBOM files are generated in SPDX format, which is a standard format for SBOMs.
