from core.gamedata import GameState


def ai(game_state: GameState) -> list[int]:
    import numpy as np

    ghost_pos = game_state.ghosts_pos
    pacman_pos = game_state.pacman_pos

    vector1 = np.array(pacman_pos) - np.array(ghost_pos[0])
    standardize1 = vector1 / np.linalg.norm(vector1)

    vector2 = np.array(pacman_pos) - np.array(ghost_pos[1])
    standardize2 = vector2 / np.linalg.norm(vector2)

    vector3 = np.array(pacman_pos) - np.array(ghost_pos[2])
    standardize3 = vector3 / np.linalg.norm(vector3)

    direction = standardize1 + standardize2 + standardize3

    if abs(direction[0]) > abs(direction[1]):
        if direction[0] > 0:
            return [4]
        else:
            return [2]
    else:
        if direction[1] > 0:
            return [1]
        else:
            return [3]
