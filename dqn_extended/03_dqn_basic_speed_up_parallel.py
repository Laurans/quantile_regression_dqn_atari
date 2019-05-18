import click
import gym
import ptan

import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import torch.multiprocessing as mp

from dqn_extended.common import configreader, neuralnetworks, trackers, losses
import wandb

torch.backends.cudnn.benchmark = True


def init_logger(params):
    wandb.init(
        name="dqn_classic_speed_up_parallel",
        project="dqn_extended",
        dir="../wandb",
        config=params,
        reinit=True,
    )


def play_func(params, net, exp_queue):
    init_logger(params)

    wandb.watch(net)

    env = gym.make(params["env_name"])
    env = ptan.common.wrappers.wrap_dqn(env)

    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=params["epsilon_start"])
    epsilon_tracker = trackers.EpsilonTracker(selector, params)
    agent = ptan.agent.DQNAgent(net, selector, device=params["device"])

    exp_source = ptan.experience.ExperienceSourceFirstLast(
        env, agent, gamma=params["gamma"], steps_count=1
    )
    exp_source_iter = iter(exp_source)

    frame_idx = 0

    with trackers.RewardTracker(params["stop_reward"]) as reward_tracker:
        while True:
            frame_idx += 1
            exp = next(exp_source_iter)
            exp_queue.put(exp)

            epsilon_tracker.frame(frame_idx)

            new_rewards = exp_source.pop_total_rewards()

            if new_rewards:
                if reward_tracker.reward(new_rewards[0], frame_idx, selector.epsilon):
                    break

    exp_queue.put(None)


@click.command()
@click.option("--gpu", default=0, type=int, help="GPU id")
def main(gpu):
    mp.set_start_method("spawn")
    params = configreader.get_config("./common/config/hyperparams.yaml")["pong"]
    params["device"] = f"cuda:{gpu}"
    params["batch_size"] *= params["train_freq"]

    env = gym.make(params["env_name"])
    env = ptan.common.wrappers.wrap_dqn(env)

    net = neuralnetworks.DQN(env.observation_space.shape, env.action_space.n)
    net = net.to(params["device"])

    tgt_net = ptan.agent.TargetNet(net)

    buffer = ptan.experience.ExperienceReplayBuffer(
        None, buffer_size=params["replay_size"]
    )
    optimizer = optim.Adam(
        net.parameters(), lr=params["learning_rate"], **params["optim_params"]
    )

    exp_queue = mp.Queue(maxsize=params["train_freq"] * 2)
    play_proc = mp.Process(target=play_func, args=(params, net, exp_queue))
    play_proc.start()

    frame_idx = 0

    while play_proc.is_alive():
        frame_idx += params["train_freq"]

        for _ in range(params["train_freq"]):
            exp = exp_queue.get()
            if exp is None:
                play_proc.join()
                break

            buffer._add(exp)

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
        clip_grad_norm_(net.parameters(), params["gradient_clip"])
        optimizer.step()

        if frame_idx % params["target_net_sync"] < params["train_freq"]:
            tgt_net.sync()


if __name__ == "__main__":
    main()
