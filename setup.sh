#!/bin/bash

pip install --upgrade pip wheel setuptools

pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

pip install flash-attn --no-build-isolation

pip install -r requirements.txt