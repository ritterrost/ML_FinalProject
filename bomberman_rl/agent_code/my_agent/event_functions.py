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


def walked_towards_closest_coin(state, action):
    if (np.array(A_TO_COORD[action]) == state[0:2]).all(): # align action with direction of coin
        return 1

def walked_from_danger(state, action):
    if not state[8] == 0:  # check if in danger (distance to free tile)
        if (np.array(A_TO_COORD[action]) == state[6:8]).all():  # align action with direction of free field
            return 1


def drop_bomb_next_to_crate(state, action):
    if state[5] == 1:  # distance to crate
        if action == "BOMB":
            return 1
    else:
        return 0

def has_no_escape(new_state, action):
    if action == 'BOMB':
        if new_state[8] > 4 or new_state[8] == -1:
            return 1


# def in_danger_zone(self):
#     t = self.transitions[-1]
#     if t.state[6] == 1:
#         return 1
#     else:
        # return 0
# 