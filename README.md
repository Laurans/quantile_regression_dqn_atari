# Distributional RL with Quantile Regression DQN

This project contains the code for learning a Quantile Regression DQN on Atari environment.

This implementation assume that you are using a workstation with a gpu.

## W&B for logging

We use W&B for logging of network parameters and others. For logging, please follow the steps below after requirement installation:

1. Create a wandb account
2. Check your API key in settings, and login wandb on your terminal: $ wandb login API_KEY
3. Initialize wandb: $ wandb init

For more details, read [W&B tutorial](https://docs.wandb.com/docs/started.html).

> Note: in practice, I use W&B in offline mode then synchronize the log when the run is complete.

## Getting started
### Prerequisites

This repository is tested on [poetry](http://poetry.eustace.io/) virtual environment


### Installation

```
poetry install
```

> Note: pyorch is a bit heavy to download, so I download in advances the wheel with `pip download pytorch` to set up easily several virtual environment.

### Usage

All the scripts are in the `dqn_extended` repository.

For testing a trained model, use the `00_testing_model.py` script.
```
poetry run python `00_testing_model.py` <model_path> <output_render_dir>
```

For training, you can use the other script with following cli
```
poetry run python <script> --gpu <gpu_id> 
```

Hyperparameters and config is located in `dqn_extended/common/config/hyperparams.yaml` file.