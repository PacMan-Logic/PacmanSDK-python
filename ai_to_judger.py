'''
TODO:lxy(done)
两个函数
ghost_to_judger(int op1 , int op2 , int op3) 传入3个数字，发包给judger
pacman_to_judger

'''

import json
from utils.utils import write_to_judger

def ghost_to_judger(op1: int, op2: int, op3: int):
    operation = [op1, op2, op3]
    
    action = ""
    
    for op in operation:
        assert 0 <= op <= 4, "operation must be between 0 and 4"
        action += str(op)
        action += ' '
    
    if action:
        action = action[:-1]
    
    message = {
		"role": 1,
        "action": action,
	}
    
    write_to_judger(json.dumps(message))

def pacman_to_judger(op: int):
    assert 0 <= op <= 4, "operation must be between 0 and 4"
    
    message = {
		"role": 0,
        "action": str(op),
	}
    
    write_to_judger(json.dumps(message))