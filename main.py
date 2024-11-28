import sys
import json

from ai_to_judger import pacman_to_judger
from ai_to_judger import ghost_to_judger
from ai import ai
from core.GymEnvironment import PacmanEnv
from core.gamedata import GameState
from utils.utils import write_to_judger

def pacman_op(env: PacmanEnv,ai):
    op = ai(env.game_state()) # 返回一个含1个元素的数组
    print(f"pacman send operation {op[0]}", file=sys.stderr)
    pacman_to_judger(op[0])

def ghosts_op(env: PacmanEnv,ai):
    op = ai(env.game_state()) # 返回一个含3个元素的数组
    print(f"ghosts send operation {op[0]} {op[1]} {op[2]}", file=sys.stderr)
    ghost_to_judger(op[0],op[1],op[2])

class Controller:
    def __init__(self):
        self.env = PacmanEnv()
        id = int(input())
        self.id = id
        self.level_change = True

    def run(self, ai):
        while 1:
            if self.level_change == True:
                init_info = input()
                self.env.ai_reset(json.loads(init_info))
                self.level_change = False
            if self.id == 0:
                #当前为0号玩家

                # 0号玩家发送信息
                pacman_op(self.env,ai)
                
                # 1号玩家发送信息
                get_info = input()
                print(f"receive info: {get_info}", file=sys.stderr)

                # 接收信息，调用step
                get_op = input()
                print(f"receive operation info: {get_op}", file=sys.stderr)
                get_op_json = json.loads(get_op)
                pacman_action = get_op_json["pacman_action"]
                ghosts_action = get_op_json["ghosts_action"]
                board , score , self.level_change = self.env.step(pacman_action,ghosts_action)
            else:
                #当前为1号玩家

                # 0号玩家发送信息
                get_info = input()
                print(f"receive info: {get_info}", file=sys.stderr)
                
                # 1号玩家发送信息
                pacman_op(self.env,ai)

                # 接收信息，调用step
                get_op = input()
                print(f"receive operation info: {get_op}", file=sys.stderr)
                get_op_json = json.loads(get_op)
                pacman_action = get_op_json["pacman_action"]
                ghosts_action = get_op_json["ghosts_action"]
                board , score , self.level_change = self.env.step(pacman_action,ghosts_action)


if __name__ == "__main__":
    print("init done", file=sys.stderr)
    controller = Controller()
    controller.run(ai)