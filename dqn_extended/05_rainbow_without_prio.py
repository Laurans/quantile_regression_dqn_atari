import os

import click
import gym
import ptan
import torch
import torch.optim as optim
import wandb
from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from coolname import generate

from dqn_extended.common import configreader, losses, neuralnetworks, trackers

torch.backends.cudnn.benchmark = True

os.environ["WANDB_MODE"] = "dryrun"


def init_logger(params):
    project = "dqn_extended"
    name = "dqn_prio_replay"
    logs_dir = "../logs"
    uid = "_".join([generate()[0], name])
    wandb.init(name=name, project=project, dir=logs_dir, config=params)

    writer = SummaryWriter(logs_dir + "/tensorboard/" + uid)
    return writer


def write_log(writer: SummaryWriter, scalars_dict: dict, step):
    for key, value in scalars_dict.items():
        writer.add_scalar(key, value, global_step=step)

    wandb.log(scalars_dict, step=step)


def watch_model(model: torch.nn.Module):
    wandb.watch(model, log=None)


def get_new_beta(frame_idx, beta_start, beta_frames):
    return min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)


@click.command()
@click.option("--gpu", default=0, type=int, help="GPU id")
def main(gpu):
    params = configreader.get_config("./common/config/hyperparams.yaml")["pong"]
    params["device"] = f"cuda:{gpu}"
    params["train_freq"] = 2
    params["batch_size"] *= params["train_freq"]
    writer = init_logger(params)

    env = gym.make(params["env_name"])
    env = ptan.common.wrappers.wrap_dqn(env)

    net = neuralnetworks.DuelingDQN(env.observation_space.shape, env.action_space.n)
    net = net.to(params["device"])

    watch_model(net)

    tgt_net = ptan.agent.TargetNet(net)
    selector = ptan.actions.ArgmaxActionSelector()
    agent = ptan.agent.DQNAgent(net, selector, device=params["device"])

    exp_source = ptan.experience.ExperienceSourceFirstLast(
        env, agent, gamma=params["gamma"], steps_count=params["reward_steps"]
    )
    buffer = ptan.experience.PrioritizedReplayBuffer(
        exp_source, buffer_size=params["replay_size"], alpha=params["prio_replay_alpha"]
    )
    optimizer = optim.Adam(
        net.parameters(), lr=params["learning_rate"], **params["optim_params"]
    )

    frame_idx = 0

    loss_in_float = None
    i_episode = 0
    with trackers.RewardTracker(params["stop_reward"]) as reward_tracker:
        while True:
            frame_idx += params["train_freq"]
            buffer.populate(params["train_freq"])
            beta = get_new_beta(frame_idx, params["beta_start"], params["beta_frames"])

            new_rewards = exp_source.pop_total_rewards()

            if new_rewards:
                i_episode = (i_episode + 1) % params["logging_freq"]
                success, logs = reward_tracker.reward(new_rewards[0], frame_idx)

                if loss_in_float:
                    logs["loss"] = loss_in_float

                if i_episode == 0:
                    write_log(writer, logs, frame_idx)

                if success:
                    write_log(writer, logs, frame_idx)
                    break

            if len(buffer) < params["replay_initial"]:
                continue

            optimizer.zero_grad()
            batch, batch_indices, batch_weights = buffer.sample(
                params["batch_size"], beta
            )
            loss, sample_prios = losses.calc_loss_dqn_prio_replay(
                batch,
                batch_weights,
                net,
                tgt_net.target_model,
                gamma=params["gamma"] ** params["reward_steps"],
                device=params["device"],
            )
            loss.backward()
            clip_grad_norm_(net.parameters(), params["gradient_clip"])
            optimizer.step()
            buffer.update_priorities(batch_indices, sample_prios.data.cpu().numpy())

            loss_in_float = float(loss.detach().cpu().numpy())

            if frame_idx % params["target_net_sync"] < params["train_freq"]:
                tgt_net.sync()

            if frame_idx > 1.5e6:
                break


if __name__ == "__main__":
    main()
