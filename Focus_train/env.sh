#!/usr/bin/env bash
# Example environment for running inside the BioNeMo container.

export DOCKER_IMAGE="bionemo-lora:2.6.3"
export CONTAINER_NAME="focus-train"
export HOST_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CONTAINER_PROJECT_ROOT="/workspace/focus-train"
# Used by configs/focus_config.yaml to resolve relative paths inside the container.
export PROJECT_ROOT="${CONTAINER_PROJECT_ROOT}"
