#!/bin/sh

# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

OPENCV_DEBUG=0

# A shell ignores signals without a handler when it is PID 1: a stop during the
# device provisioning or a preparation would wait for the kill timeout.
trap 'exit 143' TERM INT

# Disable core dumps: inherited by python and any native library it loads.
# Set ENABLE_CORE_DUMPS=1 to keep them (e.g. to debug a native crash).
if [ "${ENABLE_CORE_DUMPS:-0}" != "1" ]; then
  ulimit -c 0 2>/dev/null || true
fi

if [ -z "$PYTHONUNBUFFERED" ]; then
  export PYTHONUNBUFFERED=1
fi

# No user site-packages: the launcher, the apps and their venvs import from the
# image and the app venv only, never from what a home holds
export PYTHONNOUSERSITE=1

if [ "${OPENCV_DEBUG:-0}" = "1" ]; then
  export OPENCV_LOG_LEVEL="${OPENCV_LOG_LEVEL:-DEBUG}"
  export OPENCV_VIDEOIO_DEBUG="${OPENCV_VIDEOIO_DEBUG:-1}"
  export GST_DEBUG="${GST_DEBUG:-3}"
  export GST_DEBUG_NO_COLOR="${GST_DEBUG_NO_COLOR:-1}"
fi

provision_devices() {
  # Merge the host DSP payload and yaml config into /usr/share/hexagon-dsp,
  # where the fastrpc client libraries look for them. The script ships with the
  # libraries in python-base. Best-effort: a failure here only means the DSP is
  # unavailable, the app still starts.
  sh /provision-fastrpc-dsp.sh || echo "Warning: fastrpc DSP provisioning failed"
}

# serve, the default, runs the launcher; prepare is how the launcher installs
# the dependencies of the app at APP_DIR
case "${1:-serve}" in
  serve)
    # Once for every app the launcher runs
    provision_devices
    exec arduino-app-launcher
    ;;
  prepare)
    ;;
  *)
    echo "Unknown mode '$1': use serve or prepare" >&2
    exit 2
    ;;
esac

BASE_DIR="${APP_DIR:?prepare needs APP_DIR, the app folder}"
CACHE_DIR="$BASE_DIR/.cache"
APP_YAML="$BASE_DIR/app.yaml"
REQUIREMENTS_FILE="$BASE_DIR/python/requirements.txt"
PYTHON_LIBS_DIR="$BASE_DIR/python-libraries"
INSTALLED_REQUIREMENTS_FILE="$CACHE_DIR/installed_requirements.txt"

# Set when a dependency install fails: the preparation fails once all were tried.
DEPS_FAILED=0
# Set when something was installed: the uv cache is worth cleaning only then.
DEPS_INSTALLED=0

export UV_CACHE_DIR="$CACHE_DIR/uv"
# The installs follow this script and the image environment only, no uv.toml or
# pyproject.toml found in the app folder or the home steers them
export UV_NO_CONFIG=1

# Remove core dumps left in the app directory by a previous run, unless they
# are explicitly wanted
if [ "${ENABLE_CORE_DUMPS:-0}" != "1" ]; then
  find "$BASE_DIR" -maxdepth 1 -type f \
    \( -name 'core' -o -name 'core.[0-9]*' -o -name '*.core' \) \
    -print -delete 2>/dev/null || true
fi

# Installs run from the app folder, whatever the image WORKDIR
cd "$BASE_DIR"

check_streamlit_ui() {
  grep -q "arduino:streamlit_ui" "$APP_YAML"
}

has_requirements() { # <file>: exists with at least one non-blank line
  [ -f "$1" ] && [ "$(grep -c '[^[:space:]]' "$1")" -ne 0 ]
}

# The venv holds what the app adds to the image: its requirements, its wheels, the
# requirements of its custom bricks, streamlit. An app that adds nothing runs on the
# system interpreter as it is: `uv venv` costs 0.4 s on the board at every first start.
# A venv that exists is kept, whatever the app needs now.
needs_venv() {
  [ -d "$CACHE_DIR/.venv" ] && return 0
  [ -d "$PYTHON_LIBS_DIR" ] && return 0
  has_requirements "$REQUIREMENTS_FILE" && return 0
  for brick_requirements in "$BASE_DIR"/bricks/*/requirements.txt; do
    has_requirements "$brick_requirements" && return 0
  done
  check_streamlit_ui
}

mkdir -p "$CACHE_DIR"
if needs_venv; then
  if [ ! -d "$CACHE_DIR/.venv" ]; then
    uv venv "$CACHE_DIR/.venv" --system-site-packages

    if [ -d "$PYTHON_LIBS_DIR" ]; then
      echo "Installing Python libraries from $PYTHON_LIBS_DIR"
      # Iterate over each .whl.installed file in the directory and revert them to .whl
      for installed_file in "$PYTHON_LIBS_DIR"/*.whl.installed; do
        if [ -f "$installed_file" ]; then
          original_file="${installed_file%.installed}"
          echo "  Restoring: $installed_file -> $original_file"
          mv "$installed_file" "$original_file"
        fi
      done
    fi
  fi

  # Not bin/activate: it hardcodes the path the venv was created at, which differs with APP_DIR
  export VIRTUAL_ENV="$CACHE_DIR/.venv"
  export PATH="$VIRTUAL_ENV/bin:$PATH"
fi

if [ -d "$PYTHON_LIBS_DIR" ]; then
  echo "Installing Python libraries from $PYTHON_LIBS_DIR"
  # Iterate over each .whl file in the directory
  for whl_file in "$PYTHON_LIBS_DIR"/*.whl; do
    if [ -f "$whl_file" ]; then
      uv pip install --compile-bytecode "$whl_file"
      mv "$whl_file" "$whl_file".installed
      DEPS_INSTALLED=1
    fi
  done
fi

if [ -f "$REQUIREMENTS_FILE" ]; then
  INSTALL_DEPS=1
  REQUIREMENTS_LINES="$(grep -c '[^[:space:]]' "$REQUIREMENTS_FILE")"
  if [ -f "$INSTALLED_REQUIREMENTS_FILE" ]; then
    if cmp -s "$REQUIREMENTS_FILE" "$INSTALLED_REQUIREMENTS_FILE"; then
      echo "Requirements already installed."
      INSTALL_DEPS=0
    fi
  fi
  if [ "$INSTALL_DEPS" -gt 0 ]; then
    if [ "$REQUIREMENTS_LINES" -ne 0 ]; then
      if uv pip install --compile-bytecode -r "$REQUIREMENTS_FILE"; then
        cp "$REQUIREMENTS_FILE" "$INSTALLED_REQUIREMENTS_FILE"
        DEPS_INSTALLED=1
      else
        DEPS_FAILED=1
      fi
    else
      cp "$REQUIREMENTS_FILE" "$INSTALLED_REQUIREMENTS_FILE"
    fi
  fi
fi

# Install custom brick requirements with caching
if [ -d "$BASE_DIR/bricks" ]; then
  for brick_dir in $BASE_DIR/bricks/*; do
    if [ -d "$brick_dir" ]; then
      brick_name=$(basename "$brick_dir")
      brick_requirements="${brick_dir}/requirements.txt"
      brick_cache_dir="$CACHE_DIR/$brick_name"
      brick_installed_requirements="$brick_cache_dir/installed_requirements.txt"
      
      if [ -f "$brick_requirements" ]; then
        mkdir -p "$brick_cache_dir"
        
        INSTALL_BRICK_DEPS=1
        BRICK_REQUIREMENTS_LINES="$(grep -c '[^[:space:]]' "$brick_requirements")"
        
        if [ -f "$brick_installed_requirements" ]; then
          if cmp -s "$brick_requirements" "$brick_installed_requirements"; then
            echo "Brick requirements for '$brick_name' already installed."
            INSTALL_BRICK_DEPS=0
          fi
        fi
        
        if [ "$INSTALL_BRICK_DEPS" -gt 0 ]; then
          if [ "$BRICK_REQUIREMENTS_LINES" -ne 0 ]; then
            echo "Installing requirements for brick: $brick_name"
            if uv pip install --compile-bytecode -r "$brick_requirements"; then
              cp "$brick_requirements" "$brick_installed_requirements"
              DEPS_INSTALLED=1
            else
              DEPS_FAILED=1
            fi
          else
            cp "$brick_requirements" "$brick_installed_requirements"
          fi
        fi
      fi
    fi
  done
fi

if [ "$DEPS_INSTALLED" -ne 0 ]; then
  uv cache clean
fi

set -e
# The launcher runs a Streamlit app with the Streamlit of its venv
if check_streamlit_ui && ! uv pip show streamlit > /dev/null 2>&1; then
  echo "streamlit not found, installing..."
  uv pip install --no-cache-dir --link-mode=copy --compile-bytecode streamlit
fi
if [ "$DEPS_FAILED" -ne 0 ]; then
  echo "======== App preparation failed ====================="
  exit 1
fi
# Validate the app sources, skipping the ._ resource files a macOS archive may carry: turns a syntax error into a prepare failure
# instead of a crash on the first start. The cache prefix sends the
# bytecode to a throwaway directory, leaving the app folder untouched.
PYCHECK_DIR="$(mktemp -d)"
PYCHECK_STATUS=0
if [ -d "$BASE_DIR/bricks" ]; then
  PYTHONPYCACHEPREFIX="$PYCHECK_DIR" python -m compileall -q -x '/\._' "$BASE_DIR/python" "$BASE_DIR/bricks" || PYCHECK_STATUS=$?
else
  PYTHONPYCACHEPREFIX="$PYCHECK_DIR" python -m compileall -q -x '/\._' "$BASE_DIR/python" || PYCHECK_STATUS=$?
fi
# The launcher prepares many apps in one container: no leftovers in /tmp
rm -rf "$PYCHECK_DIR"
if [ "$PYCHECK_STATUS" -ne 0 ]; then
  exit "$PYCHECK_STATUS"
fi
# What was prepared, so that the launcher prepares again only when the dependencies change: a release ships it with the venv
exec arduino-app-launcher record-preparation "$BASE_DIR"
