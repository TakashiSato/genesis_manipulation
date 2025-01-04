#!/bin/bash -e

# create venv and activate
cd ~/genesis_ws/
python3 -m venv .venv
source .venv/bin/activate

# install Genesis
cd ~/genesis_ws/
git clone https://github.com/Genesis-Embodied-AI/Genesis.git
cd Genesis
pip3 install -e .

# install rsl_rl
cd ~/genesis_ws/
git clone --depth 1 -b v1.0.2 https://github.com/leggedrobotics/rsl_rl
cd rsl_rl
pip3 install -e .

# install tensorboard
pip3 install tensorboard