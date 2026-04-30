# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import os
import sys
from setuptools.build_meta import build_wheel as _orig_build_wheel
from setuptools.build_meta import build_sdist as _orig_build_sdist
from setuptools.build_meta import build_editable as _orig_build_editable
from setuptools.build_meta import (
    get_requires_for_build_editable as _orig_get_requires_for_build_editable,
    prepare_metadata_for_build_editable as _orig_prepare_metadata_for_build_editable,
)
import subprocess
import shutil


def run_preprocessing() -> None:
    version = os.environ.get("BRICKS_RELEASE_VERSION", "0.0.0")

    cache_folder_path = "src/arduino/app_bricks/static"
    if os.path.exists(cache_folder_path) and os.path.isdir(cache_folder_path):
        shutil.rmtree(cache_folder_path)
    os.makedirs(cache_folder_path, exist_ok=True)

    print(f"################################## Provisioning bricks list and compose files (version: {version}) #################################")
    cmd = [
        "arduino-bricks-list-modules",
        "-o",
        f"{cache_folder_path}/bricks-list.yaml",
        "-p",
        "-b",
        "-c",
        cache_folder_path,
    ]
    subprocess.run(cmd, check=True, cwd=os.getcwd())

    print(f"################################## Embed models list ###############################################################################")
    shutil.copyfile("models/models-list.yaml", f"{cache_folder_path}/models-list.yaml")

    print("################################### Docs generation #################################################################################")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    print(f"Project root: {project_root}")
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    try:
        from docs_generator import runner

        runner.run_docs_generator()
    finally:
        if project_root in sys.path:
            sys.path.remove(project_root)


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    return _orig_build_wheel(wheel_directory, config_settings, metadata_directory)


def build_sdist(sdist_directory, config_settings=None):
    run_preprocessing()
    return _orig_build_sdist(sdist_directory, config_settings)


def build_editable(editable_build_directory, config_settings=None, metadata_directory=None):
    return _orig_build_editable(editable_build_directory, config_settings, metadata_directory)


def get_requires_for_build_editable(config_settings=None):
    return _orig_get_requires_for_build_editable(config_settings)


def prepare_metadata_for_build_editable(metadata_directory, config_settings=None):
    return _orig_prepare_metadata_for_build_editable(metadata_directory, config_settings)
