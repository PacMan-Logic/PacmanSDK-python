'''
TODO:lxy
两个函数
ghost_to_judger(int op1 , int op2 , int op3) 传入3个数字，发包给judger
pacman_to_judger

'''

import json
from utils.utils import write_to_judger

def create_message(operation: int) -> str:
    message = {
        "role": 0,
        "operation": operation
    }
    return json.dumps(message)

# use functions below to send your operation to judger
def stay(): 
    msg = create_message(0)
    write_to_judger(msg)

def up():
    msg = create_message(1)
    write_to_judger(msg)

def left():
    msg = create_message(2)
    write_to_judger(msg)

def down():
    msg = create_message(3)
    write_to_judger(msg)

def right():
    msg = create_message(4)
    write_to_judger(msg)