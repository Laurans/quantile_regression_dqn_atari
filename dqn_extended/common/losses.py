import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def unpack_batch(batch):
    states, actions, rewards, dones, last_states = [], [], [], [], []
    for exp in batch:
        state = np.array(exp.state, copy=False)
        states.append(state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            last_states.append(state)  # the result will be masked anyway
        else:
            last_states.append(np.array(exp.last_state, copy=False))
    return (
        np.array(states, copy=False),
        np.array(actions),
        np.array(rewards, dtype=np.float32),
        np.array(dones, dtype=np.uint8),
        np.array(last_states, copy=False),
    )


def calc_loss_dqn(batch, net, tgt_net, gamma, device):
    states, actions, rewards, dones, next_states = unpack_batch(batch)

    states = torch.tensor(states).to(device)
    next_states = torch.tensor(next_states).to(device)
    actions = torch.tensor(actions).to(device)
    rewards = torch.tensor(rewards).to(device)
    done_mask = torch.ByteTensor(dones).to(device)

    state_action_values = net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
    next_state_values = tgt_net(next_states).max(1)[0]
    next_state_values[done_mask] = 0.0

    expected_state_action_values = next_state_values.detach() * gamma + rewards
    return nn.MSELoss()(state_action_values, expected_state_action_values)


def calc_loss_double_dqn(batch, net, tgt_net, gamma, device):
    states, actions, rewards, dones, next_states = unpack_batch(batch)

    states = torch.tensor(states).to(device)
    next_states = torch.tensor(next_states).to(device)
    actions = torch.tensor(actions).to(device)
    rewards = torch.tensor(rewards).to(device)
    done_mask = torch.ByteTensor(dones).to(device)

    state_action_values = net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

    next_states_actions = net(next_states).max(1)[1].unsqueeze(-1)
    next_state_values = tgt_net(next_states).gather(1, next_states_actions).squeeze(-1)

    next_state_values[done_mask] = 0.0

    expected_state_action_values = next_state_values.detach() * gamma + rewards
    return nn.MSELoss()(state_action_values, expected_state_action_values)


def calc_loss_dqn_prio_replay(batch, batch_weights, net, tgt_net, gamma, device):
    states, actions, rewards, dones, next_states = unpack_batch(batch)

    states = torch.tensor(states).to(device)
    next_states = torch.tensor(next_states).to(device)
    actions = torch.tensor(actions).to(device)
    rewards = torch.tensor(rewards).to(device)
    done_mask = torch.ByteTensor(dones).to(device)
    batch_weights = torch.tensor(batch_weights).to(device)

    state_action_values = net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

    next_states_actions = net(next_states).max(1)[1].unsqueeze(-1)
    next_state_values = tgt_net(next_states).gather(1, next_states_actions).squeeze(-1)

    next_state_values[done_mask] = 0.0

    expected_state_action_values = next_state_values.detach() * gamma + rewards
    losses = batch_weights * (state_action_values - expected_state_action_values) ** 2
    return losses.mean(), losses + 1e-5


def calc_loss_qr(batch, batch_weights, net, tgt_net, gamma, num_quantiles, device):
    states, actions, rewards, dones, next_states = unpack_batch(batch)
    batch_size = len(batch)

    states = torch.tensor(states).to(device)
    next_states = torch.tensor(next_states).to(device)
    actions = torch.LongTensor(actions).to(device)
    rewards = torch.tensor(rewards).to(device)
    done_mask = torch.ByteTensor(dones).to(device)

    cummulative_density = (
        (2 * torch.range(num_quantiles) + 1 / (2 * num_quantiles))
        .view(1, -1)
        .to(device)
    )

    quantiles_next = tgt_net(next_states)
    a_next = net.qvals_from_quant(next_states).max(1)[1]
    quantiles_next = quantiles_next[torch.range(batch_size), a_next.data]
    quantiles_next[done_mask] = 0.0
    expected_quantiles = quantiles_next.detach() * gamma + rewards

    quantiles = net(states)[torch.range(batch_size), actions]

    print("Quantiles_next", quantiles_next.shape, "Quantiles", quantiles.shape)

    diff = None
    huber_mul = None
    huber_loss = None
    import pdb

    pdb.set_trace()
