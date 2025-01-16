import random
from collections import namedtuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from core.GymEnvironment import PacmanEnv
from model import *

env = PacmanEnv("local")

# hyperparameters
BATCH_SIZE = 8
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 10000
TARGET_UPDATE = 10
MEMORY_SIZE = 100
LEARNING_RATE = 1e-4

# initialize networks
policy_net_pacman = PacmanNet(4, 5, 40)
target_net_pacman = PacmanNet(4, 5, 40)
target_net_pacman.load_state_dict(policy_net_pacman.state_dict())
target_net_pacman.eval()

policy_net_ghost = GhostNet(4, 5, 40)
target_net_ghost = GhostNet(4, 5, 40)
target_net_ghost.load_state_dict(policy_net_ghost.state_dict())
target_net_ghost.eval()

optimizer_pacman = optim.Adam(policy_net_pacman.parameters(), lr=LEARNING_RATE)
optimizer_ghost = optim.Adam(policy_net_ghost.parameters(), lr=LEARNING_RATE)
memory = []

Transition = namedtuple(
    "Transition",
    (
        "state",
        "extra",
        "action1",
        "action2",
        "next_state",
        "next_extra",
        "reward1",
        "reward2",
    ),
)


# epsilon-greedy policy for rollout
def select_action_ghost(state, extra, epsilon, policy_net):
    if np.random.rand() < epsilon:
        return np.random.randint(size=3, low=0, high=4)
    else:
        with torch.no_grad():
            values = policy_net(state, extra).reshape(-1, 5)
            return torch.argmax(values, dim=0).cpu().numpy()


def select_action_pacman(state, extra, epsilon, policy_net):
    if np.random.rand() < epsilon:
        return np.random.randint(low=0, high=4)
    else:
        with torch.no_grad():
            return torch.argmax(policy_net(state, extra)).cpu().item()


# trainsform state dict to state tensor
def state_dict_to_tensor(state_dict):
    board = state_dict["board"]
    if isinstance(board, list):
        board = np.array(board)
    size = board.shape[0]
    # print(board)
    # pad board to 38x38
    padding_num = 38 - size
    board = np.pad(board, pad_width=(0, padding_num),
                   mode="constant", constant_values=0)
    # pacman position matrix
    pacman_pos = np.zeros((38, 38))
    if "pacman_pos" in state_dict:
        pacman_pos[state_dict["pacman_pos"][0] + padding_num][
            state_dict["pacman_pos"][1] + padding_num
        ] = 1

    # ghost position matrix
    ghost_pos = np.zeros((38, 38))
    if "ghost_pos" in state_dict:
        for ghost in state_dict["ghost_pos"]:
            ghost_pos[ghost[0] + padding_num][ghost[1] + padding_num] = 1

    portal_pos = np.zeros((38, 38))
    if "portal" in state_dict:
        portal = state_dict["portal"]
        if portal[0] != -1 and portal[1] != -1:
            portal_pos[portal[0] + padding_num][portal[1] + padding_num] = 1

    level = state_dict["level"]
    if "round" in state_dict:
        round = state_dict["round"]
    else:
        round = 0
    # board_size = state_dict['board_size']
    portal_available = False
    if "portal_available" in state_dict:
        portal_available = int(state_dict["portal_available"])

    # print(board.shape, pacman_pos.shape, ghost_pos.shape,
    #       board_area.shape, portal_pos.shape)
    return torch.tensor(
        np.stack([board, pacman_pos, ghost_pos, portal_pos]),
        dtype=torch.float32,
    ).unsqueeze(0), torch.tensor(
        [level, round, size, portal_available] * 10, dtype=torch.float32
    ).unsqueeze(
        0
    )


# optimization of the model
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = random.sample(memory, BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    state_batch = torch.cat(batch.state)
    extra_batch = torch.cat(batch.extra)
    action1_batch = torch.cat(batch.action1)
    action2_batch = torch.cat(batch.action2)
    reward1_batch = torch.cat(batch.reward1)
    reward2_batch = torch.cat(batch.reward2)
    next_state_batch = torch.cat(batch.next_state)
    next_extra_batch = torch.cat(batch.next_extra)

    # print(state_batch.shape, extra_batch.shape, action1_batch.shape, action2_batch.shape,
    #       reward1_batch.shape, reward2_batch.shape, next_state_batch.shape, next_extra_batch.shape)

    state_action_values1 = policy_net_pacman(state_batch, extra_batch).gather(
        1, action1_batch
    )
    state_action_values2 = (
        policy_net_ghost(state_batch, extra_batch)
        .gather(2, action2_batch.transpose(2, 1))
    )

    # print(state_action_values1.shape, state_action_values2.shape)

    next_state_values1 = (
        target_net_pacman(next_state_batch, next_extra_batch).max(1)[0].detach()
    )
    next_state_values2 = (
        target_net_ghost(next_state_batch, next_extra_batch).max(2)[0].detach()
    )

    # print(next_state_values1.shape, next_state_values2.shape)
    # print(reward1_batch.shape, reward2_batch.shape)

    expected_state_action_values1 = (
        next_state_values1 * GAMMA) + reward1_batch
    expected_state_action_values2 = (
        next_state_values2 * GAMMA) + reward2_batch

    # print(expected_state_action_values1.shape,
    #       expected_state_action_values2.shape)

    loss1 = F.smooth_l1_loss(
        state_action_values1, expected_state_action_values1.unsqueeze(1)
    )
    loss2 = F.smooth_l1_loss(
        state_action_values2, expected_state_action_values2.unsqueeze(2)
    )

    # print(f"{loss1=}, {loss2=}")

    optimizer_pacman.zero_grad()
    loss1.backward()
    optimizer_pacman.step()

    optimizer_ghost.zero_grad()
    loss2.backward()
    optimizer_ghost.step()


# training iteration
if __name__ == "__main__":
    num_episodes = 1000
    epsilon = EPSILON_START
    for episode in range(num_episodes):
        state = env.reset(mode="local")
        state, extra = state_dict_to_tensor(state)
        # print(state.shape, extra.shape)

        for t in range(1000):
            action1 = select_action_pacman(state, extra, epsilon, policy_net_pacman)
            action2 = select_action_ghost(state, extra, epsilon, policy_net_ghost)
            # print(action1, action2)
            next_state, reward1, reward2, done, _ = env.step(action1, action2)
            env.render('local')
            next_state, next_extra = state_dict_to_tensor(next_state)
            # next_state = torch.tensor(
            # next_state, dtype=torch.float32).unsqueeze(0)
            reward1 = torch.tensor([reward1], dtype=torch.float32)
            reward2 = torch.tensor([reward2], dtype=torch.float32)
            # print(next_state.shape, next_extra.shape)
            print(reward1, reward2)


            memory.append(
                Transition(
                    state,
                    extra,
                    torch.tensor([[action1]]),
                    torch.tensor([[action2]]),
                    next_state,
                    next_extra,
                    reward1,
                    reward2,
                )
            )
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
