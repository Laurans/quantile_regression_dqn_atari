import os
import gym
import torch
from pathlib import Path
import cv2
import click
import numpy as np
import altair as alt
import pandas as pd

from dqn_extended.common import neuralnetworks, wrappers, configreader


class SaveObservation:
    def __init__(self, directory):
        self.directory_renders = Path(directory).joinpath("renders")
        self.directory_cdfs = Path(directory).joinpath("cdfs")
        os.makedirs(self.directory_renders, exist_ok=True)
        os.makedirs(self.directory_cdfs, exist_ok=True)

        self.counter = 0
        self.action_meaning = ["NOOP", "UP", "DOWN"]

    def cdf_plot(self, best_action, actions_dist):
        actions_dist = actions_dist[0]
        data = {k: actions_dist[i] for i, k in enumerate(self.action_meaning)}
        data["Probability Space"] = np.arange(0, 1, 1 / 51) + 0.5 / 51
        source = (
            pd.DataFrame(data)
            .melt("Probability Space")
            .rename(columns={"variable": "Action", "value": "Space of Returns"})
        )

        chart = (
            alt.Chart(source)
            .mark_line()
            .encode(
                alt.X("Space of Returns"),
                alt.Y("Probability Space", scale=alt.Scale(domain=(0, 1))),
                color="Action",
            )
            .properties(title=f"Best action: {self.action_meaning[best_action]}")
        )
        return chart

    def save(self, render, best_action, actions_dist):
        cv2.imwrite(
            str(self.directory_renders.joinpath(f"{self.counter:05d}.png")),
            cv2.cvtColor(render, cv2.COLOR_RGB2BGR),
        )

        chart = self.cdf_plot(best_action, actions_dist)
        frame_name = self.directory_cdfs.joinpath(f"{self.counter:05d}.png")
        chart.save(str(frame_name), webdriver="firefox")

        self.counter += 1


@click.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path(exists=False))
def main(model_path, output_dir):
    gpu = 0
    params = configreader.get_config("./common/config/hyperparams.yaml")["pong"]
    params["device"] = f"cuda:{gpu}"

    env = gym.make(params["env_name"])
    env = wrappers.wrap_dqn(env, max_episode_steps=params["max_steps_per_episode"])
    assert isinstance(env, gym.Env)

    net: neuralnetworks.QRDQN = neuralnetworks.QRDQN(
        env.observation_space.shape, env.action_space.n, params["n_quantiles"]
    )
    net = net.to(params["device"])
    net.load_state_dict(torch.load(model_path))
    net.eval()

    log = SaveObservation(output_dir)

    for i_episode in range(1):
        print(f"Episode {i_episode}")
        obs = env.reset()
        while True:
            obs = np.asarray(obs)[None, :]
            obs = torch.from_numpy(obs).to(params["device"])
            with torch.no_grad():
                quantiles = net(obs)
                qvals = net.qvals_from_quant(quantiles)
                best_action = np.argmax(qvals.cpu().numpy())

            render = env.render(mode="rgb_array")
            log.save(render, best_action, quantiles.cpu().numpy())

            next_obs, reward, done, _ = env.step(best_action)
            obs = next_obs

            if done:
                break


if __name__ == "__main__":
    main()
