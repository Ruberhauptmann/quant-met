#!/bin/bash
set -e

poetry config repositories.gitlab "${CI_API_V4_URL}/projects/${CI_PROJECT_ID}/packages/pypi"
pip install quant-met --extra-index-url "${CI_API_V4_URL}/projects/${CI_PROJECT_ID}/packages/pypi/simple"

current_version=$(poetry version)
registry_version=$(pip show quant-met | grep "Version: " | awk '{print $2}')

if [[ "$current_version" == "$registry_version" ]];
then
  echo "Version is not bumped!"
  exit 1
fi
