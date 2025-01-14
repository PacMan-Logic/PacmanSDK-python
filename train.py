import random
from collections import namedtuple

import gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from core.GymEnvironment import PacmanEnv
from model import *

env = PacmanEnv('local')

# hyperparameters
BATCH_SIZE = 32
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 10000
TARGET_UPDATE = 10
MEMORY_SIZE = 10000
LEARNING_RATE = 1e-4

# initialize networks
policy_net_pacman = PacmanNet(5, 5)
target_net_pacman = PacmanNet(5, 5)
target_net_pacman.load_state_dict(policy_net_pacman.state_dict())
target_net_pacman.eval()

policy_net_ghost = GhostNet(5, 5)
target_net_ghost = GhostNet(5, 5)
target_net_ghost.load_state_dict(policy_net_ghost.state_dict())
target_net_ghost.eval()

optimizer_pacman = optim.Adam(
    policy_net_pacman.parameters(), lr=LEARNING_RATE)
optimizer_ghost = optim.Adam(
    policy_net_ghost.parameters(), lr=LEARNING_RATE)
memory = []

Transition = namedtuple('Transition', ('state', 'action1',
                        'action2', 'next_state', 'reward1', 'reward2'))


# epsilon-greedy policy for rollout
def select_action_ghost(state, epsilon, policy_net):
    if np.random.rand() < epsilon:
        return np.random.randint(size=3, high=4)
    else:
        with torch.no_grad():
            values = policy_net(state).reshape(-1, 5)
            return torch.argmax(values, dim=0).cpu().numpy()


def select_action_pacman(state, epsilon, policy_net):
    if np.random.rand() < epsilon:
        return np.random.randint(size=5, high=4)
    else:
        with torch.no_grad():
            return torch.argmax(policy_net(state)).cpu().item()


# trainsform state dict to state tensor
def state_dict_to_tensor(state_dict):
    size = state_dict['board_size']
    board = state_dict['board']
    # pad board to 38x38
    padding_num = (38-size)//2
    board = np.pad(board, pad_width=padding_num,
                   mode='constant', constant_values=0)
    # pacman position matrix
    pacman_pos = np.zeros((38, 38))
    pacman_pos[state_dict['pacman_pos'][0] +
               padding_num][state_dict['pacman_pos'][1]+padding_num] = 1

    # ghost position matrix
    ghost_pos = np.zeros((38, 38))
    for ghost in state_dict['ghost_pos']:
        ghost_pos[ghost[0]+padding_num][ghost[1]+padding_num] = 1

    # board area matrix
    board_area = np.ones((size, size))
    board_area = np.pad(board_area, pad_width=padding_num,
                        mode='constant', constant_values=0)

    portal_pos = np.zeros((38, 38))
    portal = state_dict['portal']
    if portal[0] != -1 and portal[1] != -1:
        portal_pos[portal[0]+padding_num][portal[1]+padding_num] = 1

    level = state_dict['level']
    round = state_dict['round']
    board_size = state_dict['board_size']
    portal_available = int(state_dict['portal_available'])

    return torch.tensor(np.stack([board, pacman_pos, ghost_pos, board_area, portal_pos]), dtype=torch.float32).unsqueeze(0), torch.tensor([level, round, board_size, portal_available]*10, dtype=torch.float32).unsqueeze(0)


# optimization of the model
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = random.sample(memory, BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    state_batch = torch.cat(batch.state)
    action1_batch = torch.cat(batch.action1)
    action2_batch = torch.cat(batch.action2)
    reward1_batch = torch.cat(batch.reward1)
    reward2_batch = torch.cat(batch.reward2)
    next_state_batch = torch.cat(batch.next_state)

    state_action_values1 = policy_net_pacman(
        state_batch).gather(1, action1_batch)
    state_action_values2 = policy_net_ghost(
        state_batch).gather(1, action2_batch)

    next_state_values1 = target_net_pacman(
        next_state_batch).max(1)[0].detach()
    next_state_values2 = target_net_ghost(
        next_state_batch).max(1)[0].detach()
    expected_state_action_values1 = (
        next_state_values1 * GAMMA) + reward1_batch
    expected_state_action_values2 = (
        next_state_values2 * GAMMA) + reward2_batch

    loss1 = F.smooth_l1_loss(state_action_values1,
                             expected_state_action_values1.unsqueeze(1))
    loss2 = F.smooth_l1_loss(state_action_values2,
                             expected_state_action_values2.unsqueeze(1))

    optimizer_pacman.zero_grad()
    loss1.backward()
    optimizer_pacman.step()

    optimizer_ghost.zero_grad()
    loss2.backward()
    optimizer_ghost.step()


# training iteration
num_episodes = 1000
epsilon = EPSILON_START
for episode in range(num_episodes):
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    for t in range(1000):
        action1 = select_action_pacman(
            state, epsilon, policy_net_pacman)
        action2 = select_action_ghost(
            state, epsilon, policy_net_ghost)
        next_state, (reward1, reward2), done, _ = env.step((action1, action2))
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        reward1 = torch.tensor([reward1], dtype=torch.float32)
        reward2 = torch.tensor([reward2], dtype=torch.float32)

        memory.append(Transition(state, torch.tensor([[action1]]), torch.tensor(
            [[action2]]), next_state, reward1, reward2))
        if len(memory) > MEMORY_SIZE:
            memory.pop(0)

        state = next_state

        optimize_model()

        if done:
            break

    if episode % TARGET_UPDATE == 0:
        target_net_pacman.load_state_dict(policy_net_pacman.state_dict())
        target_net_ghost.load_state_dict(policy_net_ghost.state_dict())

    epsilon = max(EPSILON_END, EPSILON_START - episode / EPSILON_DECAY)
