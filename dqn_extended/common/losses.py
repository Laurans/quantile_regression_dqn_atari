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


def calc_loss_rainbow_dqn(
    batch, batch_weights, net, tgt_net, vmin, vmax, n_atoms, gamma, device
):
    states, actions, rewards, dones, next_states = unpack_batch(batch)

    batch_size = len(batch)

    states = torch.tensor(states).to(device)
    next_states = torch.tensor(next_states).to(device)
    actions = torch.tensor(actions).to(device)
    batch_weights = torch.tensor(batch_weights).to(device)

    # calc at once both next and cur states
    distr, qvals = net.both(torch.cat((states, next_states)))
    next_qvals = qvals[batch_size:]
    distr = distr[:batch_size]

    next_actions = next_qvals.max(1)[1]
    next_distr = tgt_net(next_states)
    next_best_distr = next_distr[range(batch_size), next_actions.data]
    next_best_distr = tgt_net.apply_softmax(next_best_distr)
    next_best_distr = next_best_distr.data.cpu().numpy()

    dones = dones.astype(np.bool)

    # project our distribution using Bellman update
    proj_distr = distr_projection(
        next_best_distr, rewards, dones, vmin, vmax, n_atoms, gamma
    )

    # calculate net output
    state_actions_values = distr[range(batch_size), actions.data]
    state_log_sm = F.log_softmax(state_actions_values, dim=1)
    proj_distr = torch.tensor(proj_distr).to(device)

    loss = -state_log_sm * proj_distr
    loss = batch_weights * loss.sum(dim=1)
    return loss.mean(), loss + 1e-5


def distr_projection(next_distr, rewards, dones, Vmin, Vmax, n_atoms, gamma):
    """
    Perform distribution projection aka Catergorical Algorithm from the
    "A Distributional Perspective on RL" paper
    """
    batch_size = len(rewards)
    proj_distr = np.zeros((batch_size, n_atoms), dtype=np.float32)
    delta_z = (Vmax - Vmin) / (n_atoms - 1)
    for atom in range(n_atoms):
        tz_j = np.minimum(
            Vmax, np.maximum(Vmin, rewards + (Vmin + atom * delta_z) * gamma)
        )
        b_j = (tz_j - Vmin) / delta_z
        l = np.floor(b_j).astype(np.int64)
        u = np.ceil(b_j).astype(np.int64)
        eq_mask = u == l
        proj_distr[eq_mask, l[eq_mask]] += next_distr[eq_mask, atom]
        ne_mask = u != l
        proj_distr[ne_mask, l[ne_mask]] += (
            next_distr[ne_mask, atom] * (u - b_j)[ne_mask]
        )
        proj_distr[ne_mask, u[ne_mask]] += (
            next_distr[ne_mask, atom] * (b_j - l)[ne_mask]
        )
    if dones.any():
        proj_distr[dones] = 0.0
        tz_j = np.minimum(Vmax, np.maximum(Vmin, rewards[dones]))
        b_j = (tz_j - Vmin) / delta_z
        l = np.floor(b_j).astype(np.int64)
        u = np.ceil(b_j).astype(np.int64)
        eq_mask = u == l
        eq_dones = dones.copy()
        eq_dones[dones] = eq_mask
        if eq_dones.any():
            proj_distr[eq_dones, l[eq_mask]] = 1.0
        ne_mask = u != l
        ne_dones = dones.copy()
        ne_dones[dones] = ne_mask
        if ne_dones.any():
            proj_distr[ne_dones, l[ne_mask]] = (u - b_j)[ne_mask]
            proj_distr[ne_dones, u[ne_mask]] = (b_j - l)[ne_mask]
    return proj_distr
