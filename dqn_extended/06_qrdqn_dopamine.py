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

from dqn_extended.common import configreader, losses, neuralnetworks, trackers, wrappers

torch.backends.cudnn.benchmark = True

os.environ["WANDB_MODE"] = "dryrun"


class Logger:
    def init_logger(self, params):
        project = "dqn_extended"
        name = "qrdqn_dopamine_space_invaders"
        logs_dir = "../logs"
        uid = "_".join([generate()[0], name])
        print("Exp name", uid)
        wandb.init(
            name=name, project=project, dir=logs_dir, config=params, tensorboard=True
        )

        writer = SummaryWriter(logs_dir + "/tensorboard/" + uid)
        self.uid = uid
        self.logs_dir = logs_dir
        path = self.logs_dir + f"/models/{self.uid}/"
        os.makedirs(path)
        return writer

    def write_log(self, writer: SummaryWriter, scalars_dict: dict, step):
        for key, value in scalars_dict.items():
            writer.add_scalar(key, value, global_step=step)

        wandb.log(scalars_dict, step=step)

    def watch_model(self, model: torch.nn.Module):
        wandb.watch(model, log=None)

    def save_model(self, model, step):
        path = self.logs_dir + f"/models/{self.uid}/"
        torch.save(model.state_dict(), path + f"{step}.pth")


@click.command()
@click.option("--gpu", default=0, type=int, help="GPU id")
def main(gpu):
    params = configreader.get_config("./common/config/hyperparams.yaml")["pong"]
    params["device"] = f"cuda:{gpu}"
    params["batch_size"] *= params["train_freq"]
    params["learning_rate"] *= params["train_freq"]
    params["optim_params"]["eps"] *= params["train_freq"]
    params["optim_params"]["weight_decay"] *= params["train_freq"]
    logger = Logger()
    writer = logger.init_logger(params)

    env = gym.make(params["env_name"])
    env = wrappers.wrap_dqn(env, max_episode_steps=params["max_steps_per_episode"])

    net = neuralnetworks.QRDQN(
        env.observation_space.shape, env.action_space.n, params["n_quantiles"]
    )
    net = net.to(params["device"])

    logger.watch_model(net)

    tgt_net = ptan.agent.TargetNet(net)
    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=params["epsilon_start"])
    epsilon_tracker = trackers.EpsilonTracker(selector, params)
    agent = ptan.agent.DQNAgent(net.qvals, selector, device=params["device"])

    exp_source = ptan.experience.ExperienceSourceFirstLast(
        env, agent, gamma=params["gamma"], steps_count=params["reward_steps"]
    )
    buffer = ptan.experience.ExperienceReplayBuffer(
        exp_source, buffer_size=params["replay_size"]
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
            epsilon_tracker.frame(frame_idx)

            new_rewards = exp_source.pop_total_rewards()

            if new_rewards:
                i_episode = (i_episode + 1) % params["logging_freq"]
                success, logs = reward_tracker.reward(
                    new_rewards[0], frame_idx, selector.epsilon
                )

                if loss_in_float:
                    logs["loss"] = loss_in_float

                if i_episode == 0:
                    logger.write_log(writer, logs, frame_idx)

                if i_episode % 200 == 0:
                    logger.save_model(net, frame_idx)

                if success:
                    logger.write_log(writer, logs, frame_idx)
                    logger.save_model(net, frame_idx)
                    break

            if len(buffer) < params["replay_initial"]:
                continue

            optimizer.zero_grad()
            batch = buffer.sample(params["batch_size"])
            loss, _ = losses.calc_loss_qr(
                batch,
                1,
                net,
                tgt_net.target_model,
                gamma=params["gamma"] ** params["reward_steps"],
                num_quantiles=params["n_quantiles"],
                device=params["device"],
            )
            loss.backward()
            clip_grad_norm_(net.parameters(), params["gradient_clip"])
            optimizer.step()

            loss_in_float = float(loss.detach().cpu().numpy())

            if frame_idx % params["target_net_sync"] < params["train_freq"]:
                tgt_net.sync()

            if i_episode > 200:
                break


if __name__ == "__main__":
    main()
