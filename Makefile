# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

SHELL:=/bin/bash

CONDA_ENV_NAME := quant-met-dev

# Variables to test the conda environment
ifeq (,$(shell which conda))
	HAS_CONDA=False
else
	HAS_CONDA=True
	ENV_DIR=$(shell conda info --base)
	MY_ENV_DIR=$(ENV_DIR)/envs/$(CONDA_ENV_NAME)
	CONDA_ACTIVATE=. $$(conda info --base)/etc/profile.d/conda.sh ; conda activate
endif

.PHONY: help
help: # Show help for each of the Makefile recipes.
	@grep -E '^[a-zA-Z0-9 -]+:.*#'  Makefile | sort | while read -r l; do printf "\033[1;32m$$(echo $$l | cut -f 1 -d':')\033[00m:$$(echo $$l | cut -f 2- -d'#')\n"; done


environment: # Install the development environment.
ifeq (True,$(HAS_CONDA))
ifneq ("$(wildcard $(MY_ENV_DIR))","") # check if the directory is there
	@echo ">>> Found $(CONDA_ENV_NAME) environment in $(MY_ENV_DIR)."
	conda env update --file=environment.yml -n $(CONDA_ENV_NAME) --prune
else
	@echo ">>> Detected conda, but $(CONDA_ENV_NAME) is missing in $(ENV_DIR). Installing ..."
	conda env create --file=environment.yml -n $(CONDA_ENV_NAME)
endif
	git clone https://github.com/krivenko/edipack2triqs.git
	$(CONDA_ACTIVATE) $(CONDA_ENV_NAME) && cd edipack2triqs && pip install .
	rm -rf edipack2triqs
	$(CONDA_ACTIVATE) $(CONDA_ENV_NAME) && pip install -e .
	$(CONDA_ACTIVATE) $(CONDA_ENV_NAME) && pre-commit install
	@echo ">>> Everything installed, use 'conda activate $(CONDA_ENV_NAME)' to use the environment."
else
	@echo ">>> Install conda first."
	exit
endif
