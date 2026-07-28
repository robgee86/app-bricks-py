# Build definition for every container under containers/.
#
# Containers deriving from another container of this repository declare the
# dependency in their Dockerfile (FROM ${REGISTRY}app-bricks/<parent>:${BASE_IMAGE_VERSION})
# and link it here with parent_context(): bake then builds the parent in-graph,
# in dependency order, however deep the chain.
#
#   docker buildx bake --print                     # inspect the resolved definition
#   docker buildx bake python-apps-base            # build a container and its parents
#   REGISTRY=ghcr.io/arduino/ docker buildx bake   # build them all

# Registry prefix images are published under, with a trailing slash.
variable "REGISTRY" {
  default = "ghcr.io/arduino/"
}

# Tag applied to the built images. CI sets the dev or release tag.
variable "IMAGE_TAG" {
  default = "local"
}

# When set, images are additionally tagged "${IMAGE_TAG}-${RUN_NUMBER}".
variable "RUN_NUMBER" {
  default = ""
}

# Tag the downstream Dockerfiles' FROM lines reference. Parents are built
# in-graph through the parent links, so any placeholder value works; CI sets
# it to the published tag for consistency.
variable "BASE_IMAGE_VERSION" {
  default = "local"
}

# Tag holding the build cache. Empty disables both cache import and export.
variable "CACHE_TAG" {
  default = ""
}

# Rebuild without importing the cache. The cache is still exported.
variable "SKIP_CACHE" {
  default = "false"
}

# Set by GitHub Actions; used for OCI labels.
variable "GITHUB_REPOSITORY" {
  default = "arduino/app-bricks-py"
}

variable "GITHUB_SHA" {
  default = ""
}

function "image" {
  params = [container]
  result = "${REGISTRY}app-bricks/${container}"
}

function "image_tags" {
  params = [container]
  result = compact([
    "${image(container)}:${IMAGE_TAG}",
    RUN_NUMBER == "" ? "" : "${image(container)}:${IMAGE_TAG}-${RUN_NUMBER}",
  ])
}

function "cache_from" {
  params = [container]
  result = SKIP_CACHE == "true" || CACHE_TAG == "" ? [] : ["type=registry,ref=${image(container)}:${CACHE_TAG}"]
}

function "cache_to" {
  params = [container]
  result = CACHE_TAG == "" ? [] : ["type=registry,ref=${image(container)}:${CACHE_TAG},mode=max"]
}

# Resolves the parent image reference of a downstream container's FROM line to
# the parent's bake target, so bake builds it in-graph in dependency order.
function "parent_context" {
  params = [parent]
  result = { "${image(parent)}:${BASE_IMAGE_VERSION}" = "target:${parent}" }
}

target "_common" {
  platforms = ["linux/arm64"]
  attest    = ["type=provenance,disabled=true"]
  labels = {
    "org.opencontainers.image.source"   = "https://github.com/${GITHUB_REPOSITORY}"
    "org.opencontainers.image.url"      = "https://github.com/${GITHUB_REPOSITORY}"
    "org.opencontainers.image.revision" = GITHUB_SHA
    "org.opencontainers.image.version"  = IMAGE_TAG
  }
}

# Containers whose Dockerfile starts from another container of this repository.
target "_downstream" {
  inherits = ["_common"]
  args = {
    REGISTRY           = REGISTRY
    BASE_IMAGE_VERSION = BASE_IMAGE_VERSION
  }
}

group "default" {
  targets = [
    "aihub-models-runner",
    "ei-models-runner",
    "ei-qnn-models-runner",
    "gesture-recognition-runner",
    "llamacpp-npu-runner",
    "llamacpp-runner",
    "models-downloader",
    "python-apps-base",
    "python-base",
    "python-slim",
    "qairt-common-base",
  ]
}

# Release groups: a "<prefix>/X.Y.Z" tag publishes the containers of the
# matching group. Only distributed leaf images are published; intermediate
# containers are built in-graph through the parent links and stay untagged.
group "ai" {
  targets = [
    "ei-models-runner",
    "ei-qnn-models-runner",
    "gesture-recognition-runner",
    "llamacpp-npu-runner",
    "llamacpp-runner",
  ]
}

group "release" {
  targets = [
    "models-downloader",
    "python-apps-base",
  ]
}

target "aihub-models-runner" {
  inherits   = ["_downstream"]
  context    = "containers/aihub-models-runner"
  tags       = image_tags("aihub-models-runner")
  cache-from = cache_from("aihub-models-runner")
  cache-to   = cache_to("aihub-models-runner")
  contexts   = parent_context("qairt-common-base")
}

target "ei-models-runner" {
  inherits   = ["_common"]
  context    = "containers/ei-models-runner"
  tags       = image_tags("ei-models-runner")
  cache-from = cache_from("ei-models-runner")
  cache-to   = cache_to("ei-models-runner")
}

target "ei-qnn-models-runner" {
  inherits   = ["_common"]
  context    = "containers/ei-qnn-models-runner"
  tags       = image_tags("ei-qnn-models-runner")
  cache-from = cache_from("ei-qnn-models-runner")
  cache-to   = cache_to("ei-qnn-models-runner")
}

target "gesture-recognition-runner" {
  inherits   = ["_downstream"]
  context    = "containers/gesture-recognition-runner"
  tags       = image_tags("gesture-recognition-runner")
  cache-from = cache_from("gesture-recognition-runner")
  cache-to   = cache_to("gesture-recognition-runner")
  contexts   = parent_context("aihub-models-runner")
}

target "llamacpp-npu-runner" {
  inherits   = ["_downstream"]
  context    = "containers/llamacpp-npu-runner"
  tags       = image_tags("llamacpp-npu-runner")
  cache-from = cache_from("llamacpp-npu-runner")
  cache-to   = cache_to("llamacpp-npu-runner")
  contexts   = parent_context("qairt-common-base")
  args = {
    LLAMA_CPP_URL    = "https://github.com/arduino/app-bricks-py/releases/download/llamacpp%2F20260703/llamacpp-hexagon-20260703.tar.gz"
    LLAMA_CPP_DIGEST = "sha256:7aa6b9a4877b0afc0e129f6f60c1312b9e0826077dbf27bbfcdfb078bd19000f"
  }
}

target "llamacpp-runner" {
  inherits   = ["_downstream"]
  context    = "containers/llamacpp-runner"
  tags       = image_tags("llamacpp-runner")
  cache-from = cache_from("llamacpp-runner")
  cache-to   = cache_to("llamacpp-runner")
  contexts   = parent_context("python-slim")
  args = {
    LLAMA_CPP_URL    = "https://github.com/arduino/app-bricks-py/releases/download/llamacpp%2F20260703/llamacpp-cpu-20260703.tar.gz"
    LLAMA_CPP_DIGEST = "sha256:61bf2bb702a9bd80a68c0b82e0b4fcf5debc203489aca806b4766d38fdba84e3"
  }
}

target "models-downloader" {
  inherits   = ["_downstream"]
  context    = "containers/models-downloader"
  tags       = image_tags("models-downloader")
  cache-from = cache_from("models-downloader")
  cache-to   = cache_to("models-downloader")
  contexts = merge(
    { models = "models" },
    parent_context("python-slim"),
  )
}

# The wheel context must hold the arduino wheel, built with `task build`.
target "python-apps-base" {
  inherits   = ["_downstream"]
  context    = "containers/python-apps-base"
  tags       = image_tags("python-apps-base")
  cache-from = cache_from("python-apps-base")
  cache-to   = cache_to("python-apps-base")
  contexts = merge(
    { wheel = "dist" },
    parent_context("python-base"),
  )
}

target "python-base" {
  inherits   = ["_downstream"]
  context    = "containers/python-base"
  contexts   = parent_context("python-slim")
  tags       = image_tags("python-base")
  cache-from = cache_from("python-base")
  cache-to   = cache_to("python-base")
  args = {
    OPENCV_WHL_URL                              = "https://github.com/arduino/app-bricks-py/releases/download/opencv%2F4.13.0.92-20260610/opencv_python_headless-4.13.0+1ddb20b-cp313-cp313-linux_aarch64.whl"
    OPENCV_WHL_DIGEST                           = "sha256:8d5e8319df040b93a07c91155afea515e3629fcb55510d4eee4716fc165b9c0a"
    LIBCAMERA_DEB_URL                           = "https://github.com/arduino/app-bricks-py/releases/download/libcamera%2F0.7.1-qcom4/libcamera0.7_0.7.1-1.bpo13+1qcom4_arm64.deb"
    LIBCAMERA_DEB_DIGEST                        = "sha256:958b7fb3d1a851542fddd5f54a259c685f1301945a3b10bce4fa15cf7886ef0f"
    LIBCAMERA_IPA_DEB_URL                       = "https://github.com/arduino/app-bricks-py/releases/download/libcamera%2F0.7.1-qcom4/libcamera-ipa_0.7.1-1.bpo13+1qcom4_arm64.deb"
    LIBCAMERA_IPA_DEB_DIGEST                    = "sha256:04ed476cd08acc897297f47b2a35e13fc71e459156b69be0bf7364a80bfac283"
    GSTREAMER_LIBCAMERA_DEB_URL                 = "https://github.com/arduino/app-bricks-py/releases/download/libcamera%2F0.7.1-qcom4/gstreamer1.0-libcamera_0.7.1-1.bpo13+1qcom4_arm64.deb"
    GSTREAMER_LIBCAMERA_DEB_DIGEST              = "sha256:c5c38c5ddd9689cc4dd07cc433c85c325c408a67e631499679e67a616099e4f1"
    GSTREAMER_QTIQMMFSRC_DEB_URL                = "https://github.com/robgee86/app-bricks-py/releases/download/qtiqmmfsrc%2F1.8.1/qtiqmmfsrc-1.8.1.deb"
    GSTREAMER_QTIQMMFSRC_DEB_DIGEST             = "sha256:b0b764a2f7ebf369de5cc9827c2952397d2ead757de7bee3f2c9c2f468d714a9"
    GSTREAMER_LIBGSTREAMER_PLUGINS_BASE_DEB_URL = "https://github.com/robgee86/app-bricks-py/releases/download/qtiqmmfsrc%2F1.8.1/libgstreamer-plugins-base1.0-0_1.26.2-1+deb13u1_arm64.deb"
    GSTREAMER_LIBGSTREAMER_PLUGINS_BASE_DEB_DIGEST = "sha256:a6e1aaadbac810957f5c4ce981d955c686734e85c23cbb5b89ddd33299f920c6"
  }
}

target "qairt-common-base" {
  inherits   = ["_common"]
  context    = "containers/qairt-common-base"
  tags       = image_tags("qairt-common-base")
  cache-from = cache_from("qairt-common-base")
  cache-to   = cache_to("qairt-common-base")
}

target "python-slim" {
  inherits   = ["_common"]
  context    = "containers/python-slim"
  tags       = image_tags("python-slim")
  cache-from = cache_from("python-slim")
  cache-to   = cache_to("python-slim")
}
