from core.gamedata import *


def distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def example_pacman(game_state: GameState) -> list[int]:
    import numpy as np

    ghost_pos = game_state.ghosts_pos
    pacman_pos = game_state.pacman_pos

    possible_next_blocks = []
    if game_state.board[pacman_pos[0] + 1][pacman_pos[1]] != Space.WALL.value:
        possible_next_blocks.append([pacman_pos[0] + 1, pacman_pos[1], Move.UP.value])
    if game_state.board[pacman_pos[0] - 1][pacman_pos[1]] != Space.WALL.value:
        possible_next_blocks.append([pacman_pos[0] - 1, pacman_pos[1], Move.DOWN.value])
    if game_state.board[pacman_pos[0]][pacman_pos[1] + 1] != Space.WALL.value:
        possible_next_blocks.append(
            [pacman_pos[0], pacman_pos[1] + 1, Move.RIGHT.value]
        )
    if game_state.board[pacman_pos[0]][pacman_pos[1] - 1] != Space.WALL.value:
        possible_next_blocks.append([pacman_pos[0], pacman_pos[1] - 1, Move.LEFT.value])

    res = max(
        possible_next_blocks,
        key=lambda x: distance(x, ghost_pos[0])
        + distance(x, ghost_pos[1])
        + distance(x, ghost_pos[2]),
    )
    return [res[2]]


ai_func = example_pacman
