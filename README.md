# genesis_manipulation

## About

A motion planning example for a Panda arm using Genesis and reinforcement learning.

![eval](https://raw.githubusercontent.com/TakashiSato/genesis_manipulation/refs/heads/main/imgs/eval.gif)

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

- `-B` is the number of parallel environments

### train with viewer

- append `--show_viewer` to the train command
- NOTE
  - visualize only the first environment
  - train will be slower but you can see the training process

```
python3 panda_train.py --max_iterations 100 -B 4096 --show_viewer
```

### train with viewer and visualize parallel

- append `--show_viewer` and `--show_parallel` to the train command
- NOTE
  - train will be more slower than the above command

```
python3 panda_train.py --max_iterations 100 -B 128 --show_viewer --show_parallel
```

![train_parallel](https://raw.githubusercontent.com/TakashiSato/genesis_manipulation/refs/heads/main/imgs/train_parallel.gif)

### visualize training states with tensorboard

1. execute panda_train.py
2. execute tensorboard in another terminal
  ```
  source ~/genesis_ws/.venv/bin/activate
  tensorboard --logdir ~/genesis_ws/genesis_manipulation/logs
  ```
3. open http://localhost:6006/ in your browser


## evaluate trained model

```
cd ~/genesis_ws/genesis_manipulation/
source ~/genesis_ws/.venv/bin/activate
python3 src/panda_eval.py --ckpt 100
```