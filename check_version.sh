#!/bin/bash

# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

set -e

pip install quant-met poetry

current_version=$(poetry version | awk '{print $2}')
registry_version=$(pip show quant-met | grep "Version: " | awk '{print $2}')

if [[ "$current_version" != "$registry_version" ]];
then
  echo "Version is bumped!"
  exit 0
else
  exit 1
fi
