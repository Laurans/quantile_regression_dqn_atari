import argparse

import click
import gym
import ptan
import torch

# torch.backends.cudnn.benchmark = False
import torch.optim as optim

from dqn_extended.common import configreader, neuralnetworks, trackers, losses
import wandb


def init_logger(params):
    wandb.init(
        name="dqn_classic", project="dqn_extended", dir="../wandb", config=params
    )


@click.command()
@click.option("--gpu", default=0, type=int, help="GPU id")
def main(gpu):
    params = configreader.get_config("./common/config/hyperparams.yaml")["pong"]
    params["device"] = f"cuda:{gpu}"
    init_logger(params)

    env = gym.make(params["env_name"])
    env = ptan.common.wrappers.wrap_dqn(env)

    net = neuralnetworks.DQN(env.observation_space.shape, env.action_space.n)
    net = net.to(params["device"])

    wandb.watch(net)

    tgt_net = ptan.agent.TargetNet(net)
    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=params["epsilon_start"])
    epsilon_tracker = trackers.EpsilonTracker(selector, params)
    agent = ptan.agent.DQNAgent(net, selector, device=params["device"])

    exp_source = ptan.experience.ExperienceSourceFirstLast(
        env, agent, gamma=params["gamma"], steps_count=1
    )
    buffer = ptan.experience.ExperienceReplayBuffer(
        exp_source, buffer_size=params["replay_size"]
    )
    optimizer = optim.Adam(net.parameters(), lr=params["learning_rate"])

    frame_idx = 0

    with trackers.RewardTracker(params["stop_reward"]) as reward_tracker:
        while True:
            frame_idx += 1
            buffer.populate(1)
            epsilon_tracker.frame(frame_idx)

            new_rewards = exp_source.pop_total_rewards()

            if new_rewards:
                if reward_tracker.reward(new_rewards[0], frame_idx, selector.epsilon):
                    break

            if len(buffer) < params["replay_initial"]:
                continue

            optimizer.zero_grad()
            batch = buffer.sample(params["batch_size"])
            loss = losses.calc_loss_dqn(
                batch,
                net,
                tgt_net.target_model,
                gamma=params["gamma"],
                device=params["device"],
            )
            loss.backward()
            optimizer.step()

            wandb.log({"loss": float(loss.cpu().numpy())}, step=frame_idx)

            if frame_idx % params["target_net_sync"] == 0:
                tgt_net.sync()


if __name__ == "__main__":
    main()
