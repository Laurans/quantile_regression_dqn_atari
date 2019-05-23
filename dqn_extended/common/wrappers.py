from ptan.common.wrappers import (
    EpisodicLifeEnv,
    NoopResetEnv,
    MaxAndSkipEnv,
    FireResetEnv,
    ProcessFrame84,
    ImageToPyTorch,
    FrameStack,
    ClippedRewardsWrapper,
)
import gym
import numpy as np


class StochasticFrameSkip(gym.Wrapper):
    # https://github.com/openai/baselines/blob/master/baselines/common/retro_wrappers.py
    def __init__(self, env, skip, stickprob):
        gym.Wrapper.__init__(self, env)
        self.n = skip
        self.stickprob = stickprob
        self.curac = None
        self.rng = np.random.RandomState()
        self.supports_want_render = hasattr(env, "supports_want_render")

    def reset(self, **kwargs):
        self.curac = None
        return self.env.reset(**kwargs)

    def step(self, ac):
        done = False
        totrew = 0
        for i in range(self.n):
            # First step after reset, use action
            if self.curac is None:
                self.curac = ac
            # First substep, delay with probability=stickprob
            elif i == 0:
                if self.rng.rand() > self.stickprob:
                    self.curac = ac
            # Second substep, new action definitely kicks in
            elif i == 1:
                self.curac = ac
            if self.supports_want_render and i < self.n - 1:
                ob, rew, done, info = self.env.step(self.curac, want_render=False)
            else:
                ob, rew, done, info = self.env.step(self.curac)
            totrew += rew
            if done:
                break
        return ob, totrew, done, info

    def seed(self, s):
        self.rng.seed(s)


class TimeLimit(gym.Wrapper):
    # https://github.com/openai/baselines/blob/master/baselines/common/wrappers.py
    def __init__(self, env, max_episode_steps=None):
        super(TimeLimit, self).__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def step(self, ac):
        observation, reward, done, info = self.env.step(ac)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
            info["TimeLimit.truncated"] = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)


def wrap_dqn(
    env,
    stack_frames=4,
    episodic_life=True,
    reward_clipping=True,
    max_episode_steps=None,
    sticky_action=False,
):
    """Apply a common set of wrappers for Atari games."""
    assert "NoFrameskip" in env.spec.id
    if episodic_life:
        env = EpisodicLifeEnv(env)
    env = NoopResetEnv(env, noop_max=30)
    if sticky_action:
        env = StochasticFrameSkip(env, skip=4, stickprob=0.25)
    else:
        env = MaxAndSkipEnv(env, skip=4)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = ProcessFrame84(env)
    env = ImageToPyTorch(env)
    env = FrameStack(env, stack_frames)
    if reward_clipping:
        env = ClippedRewardsWrapper(env)

    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps)

    return env
