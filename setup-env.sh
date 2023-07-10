#!/usr/bin/bash
#
# setup-env.sh - It sets up the neural deskew env
# -----------------------------------------------
set -e


conda_env_exists() {
    local env=$1
    conda env list | grep "$env$" >/dev/null 2>/dev/null
}


setup_env() {
    local env=$1

    conda create --name=$env python=3.10.11
    conda activate $env

    python -m pip install --upgrade pip
    python -m pip install -r requirements-develop.txt

    pre-commit install

    conda deactivate
}


env=neural-deskew

if ! conda_env_exists $env; then
    setup_env $env
fi

conda activate $env
