from core.gamedata import *
import numpy as np
from enum import Enum
from collections import deque


class AIState(Enum):
    COLLECT = "COLLECT"  # 收集普通豆子模式
    ESCAPE = "ESCAPE"  # 逃离幽灵模式
    BONUS = "BONUS"  # 收集特殊豆子模式
    GETOUT = "GETOUT"  # 逃离传送门模式


class PacmanAI:
    def __init__(self):
        self.board_size = None
        self.current_state = AIState.COLLECT
        self.path = []
        # 历史记录
        self.history = deque(maxlen=20)
        self.init_bean_num = 0

        # 动态参数
        self.GHOST_DANGER_DISTANCE = 5

        # 状态权重
        self.weights = {
            AIState.COLLECT: {"ghost": 1.0, "bean": 1.5, "bonus": 1.5},
            AIState.ESCAPE: {"ghost": 3.0, "bean": 0.8, "bonus": 1.0},
            AIState.BONUS: {"ghost": 0.8, "bean": 1.5, "bonus": 2.0},
            AIState.GETOUT: {"ghost": 1, "bean": 1, "bonus": 1},
        }

        # 特殊豆子价值
        self.bean_values = {
            3: 1.0,  # BONUS_BEAN
            4: 2.0,  # SPEED_BEAN
            5: 2.5,  # MAGNET_BEAN
            6: 3.0,  # SHIELD_BEAN
            7: 2.5,  # DOUBLE_BEAN
            8: 3.0,  # FROZE_BEAN
        }

    def count_remaining_bean(self, game_state: GameState):
        """计算剩余豆子数量"""
        cnt = 0
        for i in range(game_state.board_size):
            for j in range(game_state.board_size):
                if game_state.board[i][j] in range(2, 8):
                    cnt += 1

        return cnt

    def point_to_vector_projection_distance(self, point, vector_start, vector_end):
        """计算点到向量的投影距离"""
        vector = vector_end - vector_start
        point_vector = point - vector_start
        vector_length = np.linalg.norm(vector)

        if vector_length == 0:
            return np.linalg.norm(point_vector)

        vector_unit = vector / vector_length
        projection_length = np.dot(point_vector, vector_unit)
        projection_vector = vector_unit * projection_length
        projection_point = vector_start + projection_vector
        return np.linalg.norm(point - projection_point)

    def can_getout_before_ghosts(self, game_state: GameState):
        """判断是否能在幽灵到达前到达传送门"""
        pacman_pos = np.array(game_state.pacman_pos)
        portal_pos = np.array(game_state.portal_coord)

        dist_to_portal = np.linalg.norm(pacman_pos - portal_pos)
        ghosts_projection_dist_to_catch = [
            self.point_to_vector_projection_distance(ghost_pos, pacman_pos, portal_pos)
            for ghost_pos in game_state.ghosts_pos
        ]

        return dist_to_portal < min(ghosts_projection_dist_to_catch) - 1

    def manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def update_state(self, game_state: GameState):
        """更新游戏状态"""
        pacman_pos = np.array(game_state.pacman_pos)
        # 计算威胁程度
        ghost_distances = [
            len(
                self.a_star_search(
                    np.array(pacman_pos), np.array(ghost_pos), game_state
                )
            )
            for ghost_pos in game_state.ghosts_pos
        ]
        min_ghost_distance = min(ghost_distances)
        # 寻找特殊豆子
        special_bean = self.find_nearest_special_bean(game_state)
        # 检查是否有护盾状态
        has_shield = game_state.pacman_skill_status[Skill.SHIELD.value] > 0
        # 状态机转换逻辑
        if min_ghost_distance < self.GHOST_DANGER_DISTANCE and not has_shield:
            # 如果可以在幽灵到达前到达传送门，优先选择GETOUT
            if (
                game_state.level < 3
                and self.can_getout_before_ghosts(game_state)
                and game_state.portal_available
                and self.count_remaining_bean(game_state) < self.init_bean_num * 0.5
            ):
                self.current_state = AIState.GETOUT
            else:
                self.current_state = AIState.ESCAPE
        elif (
            game_state.level < 3
            and game_state.portal_available
            and self.count_remaining_bean(game_state) < self.init_bean_num * 0.5
        ):
            self.current_state = AIState.GETOUT
        elif special_bean and special_bean[1] < 8:
            self.current_state = AIState.BONUS
        else:
            self.current_state = AIState.COLLECT

    def find_nearest_special_bean(self, game_state):
        """寻找最近的特殊豆子"""
        pacman_pos = np.array(game_state.pacman_pos)
        special_beans = []

        for i in range(game_state.board_size):
            for j in range(game_state.board_size):
                bean_type = game_state.board[i][j]
                if bean_type in SPECIAL_BEANS_ITERATOR:  # 特殊豆子
                    pos = np.array([i, j])
                    dist = np.linalg.norm(pacman_pos - pos)
                    value = self.bean_values[bean_type]
                    score = value / (dist + 1)  # 考虑距离和价值的综合评分
                    special_beans.append((pos, dist, score))

        if special_beans:
            # 按综合评分排序
            best_bean = max(special_beans, key=lambda x: x[2])
            return (best_bean[0], best_bean[1])
        return None

    def a_star_search(self, start: np.ndarray, goal: np.ndarray, game_state: GameState):
        """A*搜索路径"""
        open_set = set()
        open_set.add(tuple(start))
        came_from = {}
        g_score = {tuple(start): 0}
        f_score = {tuple(start): self.manhattan_distance(start, goal)}

        while open_set:
            current = min(open_set, key=lambda x: f_score.get(x, float("inf")))
            if current == tuple(goal):
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path

            open_set.remove(current)
            for direction, _ in self.get_valid_moves(list(current), game_state):
                neighbor = tuple(direction)
                tentative_g_score = g_score[current] + 1
                if tentative_g_score < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.manhattan_distance(
                        neighbor, goal
                    )
                    if neighbor not in open_set:
                        open_set.add(neighbor)

        return []

    def evaluate_position(self, pos, game_state: GameState):
        """评估位置的价值"""
        pacman_pos = np.array(game_state.pacman_pos)
        weights = self.weights[self.current_state]
        ghost_distances = [
            len(self.a_star_search(pos, np.array(ghost_pos), game_state))
            for ghost_pos in game_state.ghosts_pos
        ]
        min_ghost_distance = min(ghost_distances)
        ghost_score = (-4) * weights["ghost"] / (min_ghost_distance + 1)

        # 计算周围豆子的价值
        bean_value = 0
        scan_range = 5
        for dx in range(-scan_range, scan_range + 1):
            for dy in range(-scan_range, scan_range + 1):
                new_x, new_y = int(pos[0] + dx), int(pos[1] + dy)
                if (
                    0 <= new_x < game_state.board_size
                    and 0 <= new_y < game_state.board_size
                ):
                    bean_type = game_state.board[new_x][new_y]
                    if bean_type in BEANS_ITERATOR:  # 有豆子
                        distance = abs(dx) + abs(dy)
                        if bean_type in SPECIAL_BEANS_ITERATOR:  # 特殊豆子
                            bean_value += self.bean_values[bean_type] / (distance + 1)
                        elif bean_type == Space.REGULAR_BEAN.value:  # 普通豆子
                            bean_value += 1 / (distance + 1)

        # 避免重复访问
        pos_tuple = tuple(pos)
        repeat_penalty = 0
        visit_count = self.history.count(pos_tuple)
        repeat_penalty = -8 * visit_count
        final_score = ghost_score + bean_value * weights["bean"] + repeat_penalty

        # 如果是逃离模式，确保远离幽灵
        if self.current_state == AIState.ESCAPE:
            if min_ghost_distance < self.GHOST_DANGER_DISTANCE:
                final_score -= (self.GHOST_DANGER_DISTANCE - min_ghost_distance) * 10

        # 如果是GETOUT模式，确保尽快到达传送门
        if self.current_state == AIState.GETOUT:
            portal_pos = np.array(game_state.portal_coord)
            dist_to_portal = np.linalg.norm(pos - portal_pos)
            final_score += 30 / (dist_to_portal + 1)
        return final_score

    def get_valid_moves(self, pos, game_state):
        """获取有效的移动方向"""
        moves = []
        directions = [
            (np.array(list(Update.UP.value)), Direction.UP.value),  # UP
            (np.array(list(Update.LEFT.value)), Direction.LEFT.value),  # LEFT
            (np.array(list(Update.DOWN.value)), Direction.DOWN.value),  # DOWN
            (np.array(list(Update.RIGHT.value)), Direction.RIGHT.value),  # RIGHT
        ]
        for direction, move_num in directions:
            new_pos = pos + direction
            if self.is_valid_position(new_pos, game_state):
                moves.append((new_pos, move_num))
        return moves

    def is_valid_position(self, pos, game_state: GameState):
        """检查位置是否有效"""
        x, y = int(pos[0]), int(pos[1])
        if 0 <= x < game_state.board_size and 0 <= y < game_state.board_size:
            if self.current_state != AIState.GETOUT:
                if game_state.board[x][y] == Space.PORTAL.value:
                    return False
            if game_state.board[x][y] != Space.WALL.value:
                return True
        return False

    def choose_move(self, game_state: GameState):
        """选择移动方向"""

        # 初始化
        if game_state.round == 1:
            self.init_bean_num = self.count_remaining_bean(game_state)

        self.board_size = game_state.board_size
        self.update_state(game_state)
        pacman_pos = np.array(game_state.pacman_pos)
        valid_moves = self.get_valid_moves(pacman_pos, game_state)

        # 评估每个可能的移动
        move_scores = []
        for new_pos, move_num in valid_moves:
            score = self.evaluate_position(new_pos, game_state)
            move_scores.append((score, move_num))

        # 选择最佳移动
        if move_scores:
            best_score, best_move = max(move_scores, key=lambda x: x[0])
            # 更新历史记录
            self.history.append(tuple(pacman_pos))
            return [best_move]

        return [Direction.STAY.value]  # 默认停留

# TODO: 你需要实现一个ai函数

ai_func = PacmanAI().choose_move
__all__ = ["ai_func"]
