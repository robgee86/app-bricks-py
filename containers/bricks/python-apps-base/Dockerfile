# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

ARG REGISTRY
ARG BASE_IMAGE_VERSION=latest
FROM ${REGISTRY}app-bricks/python-base:${BASE_IMAGE_VERSION}

# The compose files provisioned at runtime reference the images published with this one
ARG BASE_IMAGE_VERSION
ENV BRICKS_RELEASE_VERSION=${BASE_IMAGE_VERSION}

# The skeleton of every app home, which the launcher mirrors into <app>/.cache/home: the Streamlit theme and the ALSA
# wrappers only, none of Debian's shell defaults
RUN rm -rf /etc/skel/.[!.]* /etc/skel/*
COPY ./streamlit/config.toml /etc/skel/.streamlit/config.toml

COPY ./scripts/*.sh /
# The wheel and the lock it was built from, from the bake context filled by `task build:bricks`
COPY --from=wheel arduino*.whl pyproject.toml uv.lock /tmp/wheel/
COPY ./patches/ /tmp/patches/
# OpenCV is left out, python-base ships its own GStreamer build of it
RUN set -ex; \
    uv export --frozen --project /tmp/wheel --extra all --no-dev --no-emit-project --no-emit-package opencv-python-headless | uv pip install --system --no-cache-dir --require-hashes -r -; \
    uv pip install --system --no-cache-dir --no-deps /tmp/wheel/*.whl; \
    rm -rf /tmp/wheel; \
    # Every patch in patches/ (see its README), in name order: one that no longer applies exactly fails the build.
    # Applied before compileall, which compiles the patched sources. GNU patch comes in through dpkg: apt-get
    # install refuses to run on the packages python-base removes with --force-depends, dpkg checks only the
    # dependencies of patch
    set -- /tmp/patches/*.patch; \
    if [ -e "$1" ]; then \
        apt-get update -y; \
        (cd /tmp && apt-get download patch); \
        dpkg -i /tmp/patch_*.deb; \
        for p in "$@"; do patch -p1 --forward --fuzz=0 --no-backup-if-mismatch -d /usr/local/lib/python3.13/site-packages -i "$p"; done; \
        dpkg --purge patch; \
        apt-get dist-clean; \
        rm -rf /var/lib/apt/lists/* /var/lib/dpkg/*-old /tmp/patch_*.deb; \
    fi; \
    rm -rf /tmp/patches; \
    # The ALSA dsnoop/dmix wrappers are static: generate them once here, into the skeleton
    HOME=/etc/skel bash /provision-alsa-devices.sh; \
    # fix permissions for /home/app
    chown arduino:arduino -R /home/app; \
    chmod 777 -R /home/app; \
    # ensure permissions for /run.sh
    chmod 755 /*.sh; \
    # trim dependencies
    find /usr/local/lib/python3.13/site-packages -type d -name "tests" -exec rm -r {} +; \
    # precompile python files to .pyc to speed up startup time
    python -m compileall /usr/local/bin; \
    python -m compileall /usr/local/lib

# /app names the running app: the launcher points this link at its folder, in place of the base image folder
RUN rmdir /app && ln -s /home/app/current-app /app

USER arduino

# The launcher runs from the root, a folder nothing writes, not from a home
WORKDIR /

ENV HOME=/home/app
ENV USER=arduino
ENV PIPEWIRE_REMOTE=/run/user/1000/pipewire-0

ENTRYPOINT [ "/run.sh" ]
