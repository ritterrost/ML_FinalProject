"""
  Input: self, events, old_game_state, new_game_state
  Output: Amount of reward: int
"""

import numpy as np
import events as e

A_TO_COORD = {
    "UP": [0, -1],
    "LEFT": [-1, 0],
    "RIGHT": [1, 0],
    "DOWN": [0, 1],
    "WAIT": [0, 0],
    "BOMB": [-1, -1],
}


# def made_suggested_move(state, action):
#     if (np.array(A_TO_COORD[action]) == state[0:2]).all():
#         return 1


def drop_bomb_next_to_crate(state, action):
    if 1 in state[0:4]:
        if action == "BOMB":
            return 1
    else:
        return 0


# def in_danger_zone(self):
#     t = self.transitions[-1]
#     if t.state[6] == 1:
#         return 1
#     else:
        # return 0
# 