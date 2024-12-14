from core.gamedata import *
import numpy as np

example_gamestate = GameState(
    level=1,
    round=1,
    pacman_score=0,
    ghosts_score=0,
    pacman_skill_status=[0, 0, 0, 0],
    pacman_pos=[36, 1],
    ghosts_pos=[[36, 36], [1, 1], [1, 36]],
    board_size=38,
    board=[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 2, 1, 5, 2, 1, 2, 2, 2, 2, 7, 1, 6, 2, 2, 2, 3, 2, 2, 2, 2, 5, 1, 2, 1, 2, 5, 1, 2, 2, 1, 1, 2, 1, 3, 2, 0], [0, 1, 2, 2, 2, 0, 2, 4, 2, 2, 3, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 2, 1, 2, 1, 1, 2, 1, 1, 1, 2, 0], [0, 2, 1, 6, 2, 0, 1, 1, 5, 6, 2, 1, 2, 2, 0, 7, 3, 7, 2, 2, 1, 2, 4, 2, 1, 2, 0, 1, 7, 0, 0, 0, 5, 2, 4, 1, 2, 0], [0, 1, 2, 1, 1, 0, 2, 2, 2, 2, 2, 7, 1, 2, 0, 3, 3, 1, 2, 1, 2, 0, 2, 2, 1, 3, 0, 2, 2, 0, 1, 5, 1, 2, 5, 7, 2, 0], [0, 6, 0, 0, 0, 0, 0, 0, 0, 2, 6, 0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 2, 1, 4, 2, 0, 0, 1, 2, 0, 2, 1, 2, 3, 2, 1, 2, 0], [0, 5, 6, 1, 1, 0, 2, 3, 2, 2, 2, 7, 1, 1, 0, 2, 2, 2, 2, 3, 2, 1, 2, 0, 2, 2, 0, 2, 2, 0, 2, 5, 2, 6, 0, 5, 2, 0], [0, 2, 5, 2, 7, 0, 4, 4, 1, 1, 2, 1, 4, 2, 0, 2, 1, 2, 2, 2, 2, 1, 1, 7, 2, 1, 0, 1, 1, 0, 1, 2, 1, 2, 1, 2, 2, 0], [0, 6, 1, 2, 1, 0, 7, 2, 6, 6, 2, 2, 3, 1, 0, 4, 2, 4, 2, 7, 1, 1, 2, 2, 2, 5, 2, 2, 5, 0, 0, 0, 0, 0, 0, 2, 2, 0], [0, 2, 6, 2, 2, 2, 2, 4, 7, 1, 1, 1, 3, 6, 1, 2, 1, 2, 2, 2, 2, 1, 2, 7, 2, 2, 2, 1, 2, 1, 2, 2, 4, 2, 3, 1, 2, 0], [0, 1, 1, 2, 2, 1, 2, 2, 2, 2, 1, 2, 2, 2, 2, 7, 2, 2, 2, 2, 2, 1, 1, 2, 1, 2, 1, 2, 1, 5, 2, 2, 1, 2, 1, 2, 2, 0], [0, 6, 0, 0, 0, 3, 0, 0, 0, 7, 2, 2, 4, 2, 0, 6, 2, 3, 2, 7, 2, 2, 2, 0, 2, 2, 6, 5, 4, 0, 0, 0, 0, 0, 0, 0, 2, 0], [0, 2, 0, 0, 7, 1, 5, 3, 0, 6, 5, 2, 1, 1, 0, 2, 5, 2, 2, 2, 7, 2, 2, 0, 4, 5, 2, 7, 7, 0, 2, 2, 1, 1, 0, 0, 2, 0], [0, 2, 0, 2, 0, 2, 7, 6, 0, 3, 3, 2, 1, 2, 0, 5, 1, 1, 2, 1, 2, 4, 2, 0, 3, 2, 2, 6, 2, 0, 2, 2, 2, 0, 2, 0, 2, 0], [0, 7, 3, 1, 4, 3, 2, 2, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 2, 6, 0, 0, 0, 0, 0, 0, 0, 2, 2, 3, 7, 7, 2, 2, 2, 3, 2, 0], [0, 1, 0, 2, 2, 2, 0, 2, 0, 2, 2, 2, 5, 1, 0, 2, 2, 5, 2, 1, 2, 4, 2, 0, 2, 1, 2, 2, 4, 0, 2, 0, 5, 3, 2, 0, 2, 0], [0, 2, 0, 2, 2, 2, 4, 0, 0, 1, 2, 2, 2, 1, 0, 1, 3, 2, 2, 2, 1, 7, 4, 0, 1, 1, 2, 1, 2, 0, 0, 2, 1, 2, 2, 0, 2, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 6, 2, 2, 2, 0, 2, 7, 2, 2, 2, 2, 2, 2, 0, 2, 1, 3, 1, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0], [0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0], [0, 1, 2, 1, 2, 6, 1, 2, 2, 5, 2, 1, 5, 1, 2, 1, 1, 2, 2, 6, 2, 1, 3, 2, 2, 2, 7, 1, 2, 2, 2, 2, 2, 1, 2, 6, 2, 0], [0, 3, 6, 2, 2, 2, 5, 6, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 2, 5, 2, 2, 2, 1, 2, 2, 6, 1, 6, 7, 6, 2, 1, 2, 2, 1, 2, 0], [0, 2, 0, 2, 2, 0, 6, 2, 1, 2, 2, 0, 2, 1, 1, 1, 0, 0, 2, 1, 0, 2, 2, 2, 2, 2, 2, 1, 2, 0, 3, 6, 1, 2, 1, 2, 2, 0], [0, 1, 0, 2, 2, 2, 2, 2, 3, 3, 1, 0, 7, 2, 3, 0, 2, 0, 2, 1, 0, 2, 2, 2, 1, 2, 2, 1, 1, 0, 0, 2, 2, 3, 2, 1, 2, 0], [0, 1, 0, 7, 4, 0, 2, 3, 6, 2, 2, 3, 2, 7, 7, 1, 2, 3, 2, 2, 0, 2, 2, 5, 0, 2, 2, 1, 3, 0, 7, 2, 2, 2, 0, 2, 2, 0], [0, 2, 0, 1, 4, 2, 6, 2, 2, 1, 2, 0, 2, 0, 2, 1, 1, 0, 2, 2, 0, 2, 2, 0, 2, 2, 2, 2, 3, 0, 0, 2, 2, 4, 1, 2, 2, 0], [0, 2, 0, 1, 2, 0, 3, 7, 7, 2, 2, 0, 0, 2, 1, 2, 2, 0, 2, 2, 0, 6, 0, 2, 2, 2, 2, 2, 2, 0, 2, 2, 6, 2, 4, 2, 2, 0], [0, 6, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 2, 2, 1, 0, 0, 0, 0, 0, 0, 1, 2, 0], [0, 2, 2, 1, 2, 6, 2, 4, 4, 7, 7, 2, 4, 2, 2, 2, 2, 6, 2, 5, 1, 2, 5, 2, 1, 2, 2, 7, 4, 2, 2, 4, 1, 1, 2, 7, 2, 0], [0, 2, 6, 1, 2, 2, 2, 2, 2, 5, 2, 2, 7, 1, 1, 1, 2, 1, 2, 1, 1, 2, 2, 1, 7, 2, 1, 1, 1, 1, 1, 2, 1, 3, 2, 2, 2, 0], [0, 2, 3, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 2, 2, 2, 1, 2, 0, 1, 4, 1, 2, 1, 1, 2, 4, 1, 1, 2, 3, 2, 0], [0, 2, 2, 1, 6, 2, 1, 0, 0, 2, 2, 1, 1, 7, 0, 7, 0, 0, 2, 2, 2, 2, 3, 0, 7, 2, 7, 1, 1, 0, 2, 2, 0, 3, 2, 2, 2, 0], [0, 3, 2, 5, 2, 1, 4, 1, 0, 6, 2, 2, 1, 0, 2, 2, 2, 0, 2, 2, 2, 2, 3, 0, 2, 2, 2, 1, 7, 0, 2, 4, 0, 1, 2, 1, 2, 0], [0, 2, 2, 2, 1, 6, 2, 2, 0, 1, 2, 2, 1, 1, 1, 1, 2, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 2, 2, 0, 1, 2, 1, 2, 0], [0, 2, 2, 6, 2, 6, 6, 0, 0, 3, 4, 2, 2, 4, 4, 1, 2, 0, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 3, 0, 2, 2, 2, 4, 2, 2, 2, 0], [0, 5, 2, 2, 2, 2, 3, 0, 0, 7, 2, 2, 2, 1, 2, 1, 2, 0, 2, 2, 2, 1, 2, 0, 2, 2, 1, 1, 1, 0, 4, 2, 1, 2, 2, 5, 2, 0], [0, 2, 2, 2, 2, 1, 2, 1, 1, 5, 6, 3, 6, 3, 6, 7, 3, 1, 2, 1, 5, 5, 3, 0, 2, 2, 2, 1, 2, 0, 0, 0, 0, 0, 0, 2, 2, 0], [0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
    portal_available=False,
    portal_coord=[10, 10],
)
import numpy as np
from enum import Enum
from collections import deque
from typing import List, Tuple

class AIState(Enum):
    COLLECT = "COLLECT"         # 收集普通豆子模式
    ESCAPE = "ESCAPE"          # 逃离幽灵模式
    BONUS = "BONUS" 
    GETOUT = "GETOUT"

class PacmanAI:
    def __init__(self):
        self.board_size = None
        self.current_state = AIState.COLLECT
        self.path = []
        self.history = deque(maxlen=20)
        
        self.init_bean_num = 0
        
        # 动态参数
        self.GHOST_DANGER_DISTANCE = 6
        self.BONUS_BEAN_PRIORITY = 2.0
        
        # 状态权重
        self.weights = {
            AIState.COLLECT: {"ghost": 1.0, "bean": 1.0, "bonus": 1.5},
            AIState.ESCAPE: {"ghost": 2.0, "bean": 0.5, "bonus": 1.0},
            AIState.BONUS: {"ghost": 0.8, "bean": 0.3, "bonus": 2.0},
            AIState.GETOUT: {"ghost": 0.5, "bean": 0.5, "bonus": 0.5}
        }
        
        # 特殊豆子价值
        self.bean_values = {
            3: 4.0,  # BONUS_BEAN
            4: 3.0,  # SPEED_BEAN
            5: 2.5,  # MAGNET_BEAN
            6: 3.0,  # SHIELD_BEAN
            7: 2.5   # DOUBLE_BEAN
        }
        
        
    def count_remaining_bean(self, game_state: GameState):
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
        ghosts_pos = np.array(game_state.ghosts_pos)
        
        dist_to_portal = np.linalg.norm(pacman_pos - portal_pos)
        ghosts_projection_dist_to_catch = [self.point_to_vector_projection_distance(ghost_pos, pacman_pos, portal_pos) for ghost_pos in ghosts_pos]
        
        return dist_to_portal < min(ghosts_projection_dist_to_catch) - 1
        

    def update_state(self, game_state: GameState):
        """更新游戏状态"""
        pacman_pos = np.array(game_state.pacman_pos)
        ghosts_pos = np.array(game_state.ghosts_pos)
        
        # 计算威胁程度
        ghost_distances = [np.linalg.norm(pacman_pos - ghost_pos) for ghost_pos in ghosts_pos]
        min_ghost_distance = min(ghost_distances)
        
        # 寻找特殊豆子
        special_bean = self.find_nearest_special_bean(game_state)
        
        # 检查是否有护盾状态
        has_shield = game_state.pacman_skill_status[3] > 0
        
        # 状态机转换逻辑
        if min_ghost_distance < self.GHOST_DANGER_DISTANCE and not has_shield:
            # 如果可以在幽灵到达前到达传送门，优先选择GETOUT
            if self.can_getout_before_ghosts(game_state) and game_state.portal_available and self.count_remaining_bean(game_state) < self.init_bean_num * 0.5:
                self.current_state = AIState.GETOUT
            else:
                self.current_state = AIState.ESCAPE
        elif game_state.portal_available and self.count_remaining_bean(game_state) < self.init_bean_num * 0.5:
            self.current_state = AIState.GETOUT
        elif special_bean and special_bean[1] < 8:
            self.current_state = AIState.BONUS
        else:
            self.current_state = AIState.COLLECT

    def find_nearest_special_bean(self, game_state) -> Tuple[np.ndarray, float]:
        """寻找最近的特殊豆子"""
        pacman_pos = np.array(game_state.pacman_pos)
        special_beans = []
        
        for i in range(game_state.board_size):
            for j in range(game_state.board_size):
                bean_type = game_state.board[i][j]
                if bean_type >= 3 and bean_type <= 7:  # 特殊豆子
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

    def evaluate_position(self, pos, game_state):
        """评估位置的价值"""
        pacman_pos = np.array(game_state.pacman_pos)
        weights = self.weights[self.current_state]
        
        # 计算与幽灵的距离
        ghost_distances = [np.linalg.norm(pos - ghost_pos) for ghost_pos in game_state.ghosts_pos]
        min_ghost_distance = min(ghost_distances)
        ghost_score = min_ghost_distance * weights["ghost"]
        
        # 计算周围豆子的价值
        bean_value = 0
        scan_range = 2
        for dx in range(-scan_range, scan_range + 1):
            for dy in range(-scan_range, scan_range + 1):
                new_x, new_y = int(pos[0] + dx), int(pos[1] + dy)
                if 0 <= new_x < game_state.board_size and 0 <= new_y < game_state.board_size:
                    bean_type = game_state.board[new_x][new_y]
                    if bean_type in range(2, 8):  # 有豆子
                        distance = abs(dx) + abs(dy)
                        if bean_type in range(3, 8):  # 特殊豆子
                            bean_value += self.bean_values[bean_type] / (distance + 1)
                        elif bean_type == 2:  # 普通豆子
                            bean_value += 1 / (distance + 1)
        
        # 避免重复访问
        pos_tuple = tuple(pos)
        repeat_penalty = 0
        if pos_tuple in self.history:
            repeat_penalty = -5
        
        final_score = ghost_score + bean_value * weights["bean"] + repeat_penalty
        
        # 如果是逃离模式，确保远离幽灵
        if self.current_state == AIState.ESCAPE:
            if min_ghost_distance < self.GHOST_DANGER_DISTANCE:
                final_score -= (self.GHOST_DANGER_DISTANCE - min_ghost_distance) * 10
                
        # 如果是GETOUT模式，确保尽快到达传送门
        if self.current_state == AIState.GETOUT:
            portal_pos = np.array(game_state.portal_coord)
            dist_to_portal = np.linalg.norm(pos - portal_pos)
            final_score -= dist_to_portal * 20
        
        return final_score

    def get_valid_moves(self, pos, game_state) -> List[Tuple[np.ndarray, int]]:
        """获取有效的移动方向"""
        moves = []
        directions = [
            (np.array([0, 0]), 0),  # STAY
            (np.array([1, 0]), 1),  # UP
            (np.array([0, -1]), 2),  # LEFT
            (np.array([-1, 0]), 3),  # DOWN
            (np.array([0, 1]), 4)   # RIGHT
        ]
        
        for direction, move_num in directions:
            new_pos = pos + direction
            if self.is_valid_position(new_pos, game_state):
                moves.append((new_pos, move_num))
        
        return moves

    def is_valid_position(self, pos, game_state) -> bool:
        """检查位置是否有效"""
        x, y = int(pos[0]), int(pos[1])
        if 0 <= x < game_state.board_size and 0 <= y < game_state.board_size:
            if self.current_state != AIState.GETOUT:
                if game_state.board[x][y] == Space.PORTAL.value:
                    return False
            return game_state.board[x][y] != 0  # 不是墙
        return False

    def choose_move(self, game_state: GameState) -> List[int]:
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
        
        return [0]  # 默认停留


ai_func = PacmanAI().choose_move
'''print(ai_func.current_state)
print(ai_func.choose_move(example_gamestate))'''
__all__ = ["ai_func"]
