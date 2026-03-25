#!/bin/bash
pip install uv
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install --upgrade pip setuptools wheel
uv pip install -r requirements.txt
uv pip install mamba-ssm==2.2.4 --no-build-isolation
uv pip install causal-conv1d==1.5.0.post8 --no-build-isolation
