# genesis_manipulation

## About

A motion planning example for a Panda arm using Genesis and reinforcement learning.

![demo](https://raw.githubusercontent.com/TakashiSato/genesis_manipulation/refs/heads/main/imgs/sample.gif)

## setup
```
git clone https://github.com/TakashiSato/genesis_manipulation.git ~/genesis_ws/genesis_manipulation
cd ~/genesis_ws/genesis_manipulation/
bash scripts/setup.bash
```

## train
```
cd ~/genesis_ws/genesis_manipulation/
source ~/genesis_ws/.venv/bin/activate
python3 src/panda_train.py --max_iterations 100 -B 4096
```

### train with viewer

- append `--show_viewer` to the train command
  - NOTE: train will be slower but you can see the training process

```
python3 panda_train.py --max_iterations 300 -B 1 --show_viewer
```

### visualize training states with tensorboard

1. execute tensorboard
  ```
  source ~/genesis_ws/.venv/bin/activate
  tensorboard --logdir ~/genesis_ws/genesis_manipulation/logs
  ```
2. open http://localhost:6006/ in your browser


## evaluate trained model

```
cd ~/genesis_ws/genesis_manipulation/
source ~/genesis_ws/.venv/bin/activate
python3 src/panda_eval.py --ckpt 300
```