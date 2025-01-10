#!/bin/bash

# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

set -e

pip install quant-met poetry

current_version=$(poetry version | awk '{print $2}')
registry_version=$(pip show quant-met | grep "Version: " | awk '{print $2}')

if [[ "$current_version" == "$registry_version" ]];
then
  echo "Version is not bumped!"
  exit 1
elif [[ "$current_version" == *"dev"* ]];
then
  echo "Version still has dev string!"
  exit 1
else
  exit 0
fi
