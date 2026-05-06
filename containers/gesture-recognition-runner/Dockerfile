# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

ARG REGISTRY
ARG BASE_IMAGE_VERSION=latest

FROM ${REGISTRY}app-bricks/aihub-models-runner:${BASE_IMAGE_VERSION}

COPY . .

RUN uv export --frozen --no-emit-project --format requirements-txt | uv pip install --python /opt/venv/bin/python3 --no-cache-dir -r - \
    && rm pyproject.toml uv.lock \
    && python -m compileall /opt/venv