import click
import gym
import ptan

import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from dqn_extended.common import configreader, neuralnetworks, trackers, losses
import wandb

torch.backends.cudnn.benchmark = True


def init_logger(params):
    wandb.init(
        name="dqn_prio_replay",
        project="dqn_extended",
        dir="../wandb",
        config=params,
    )

def get_new_beta(frame_idx, beta_start, beta_frames):
    return min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)

@click.command()
@click.option("--gpu", default=0, type=int, help="GPU id")
def main(gpu):
    params = configreader.get_config("./common/config/hyperparams.yaml")["pong"]
    params["device"] = f"cuda:{gpu}"
    params["batch_size"] *= params["train_freq"]
    init_logger(params)

    env = gym.make(params["env_name"])
    env = ptan.common.wrappers.wrap_dqn(env)

    net = neuralnetworks.DQN(env.observation_space.shape, env.action_space.n)
    net = net.to(params["device"])

    wandb.watch(net, log=None)

    tgt_net = ptan.agent.TargetNet(net)
    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=params["epsilon_start"])
    epsilon_tracker = trackers.EpsilonTracker(selector, params)
    agent = ptan.agent.DQNAgent(net, selector, device=params["device"])

    exp_source = ptan.experience.ExperienceSourceFirstLast(
        env, agent, gamma=params["gamma"], steps_count=1
    )
    buffer = ptan.experience.PrioritizedReplayBuffer(
        exp_source, buffer_size=params["replay_size"], alpha=params["prio_replay_alpha"]
    )
    optimizer = optim.Adam(
        net.parameters(), lr=params["learning_rate"], **params["optim_params"]
    )

    frame_idx = 0

    with trackers.RewardTracker(params["stop_reward"]) as reward_tracker:
        while True:
            frame_idx += params["train_freq"]
            buffer.populate(params["train_freq"])
            epsilon_tracker.frame(frame_idx)

            beta = get_new_beta(frame_idx, params["beta_start"], params["beta_frames"])

            new_rewards = exp_source.pop_total_rewards()

            if new_rewards:
                if reward_tracker.reward(new_rewards[0], frame_idx, selector.epsilon):
                    break

            if len(buffer) < params["replay_initial"]:
                continue

            optimizer.zero_grad()
            batch, batch_indices, batch_weights = buffer.sample(params["batch_size"])
            loss, sample_prios = losses.calc_loss_dqn_prio_replay(
                batch,
                batch_weights,
                net,
                tgt_net.target_model,
                gamma=params["gamma"],
                device=params["device"],
            )
            loss.backward()
            clip_grad_norm_(net.parameters(), params["gradient_clip"])
            optimizer.step()
            buffer.update_priorities(batch_indices, sample_prios.data.cpu().numpy())

            if frame_idx % params["target_net_sync"] < params["train_freq"]:
                tgt_net.sync()


if __name__ == "__main__":
    main()
