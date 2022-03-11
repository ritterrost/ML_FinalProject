"""
  Input: self, events, old_game_state, new_game_state
  Output: Amount of reward: int
"""

import numpy as np
import events as e

A_TO_COORD = {
    "RIGHT": [-1, 0],
    "UP": [0, 1],
    "DOWN": [0, -1],
    "LEFT": [1, 0],
    "WAIT": [0, 0],
    "BOMB": [0, 0],
}


def walked_towards_closest_coin(self):
    t = self.transitions[-1]
    if t.action == "BOMB" or "WAIT":
        return 0
    if (np.array(A_TO_COORD[t.action]) == t.state[4:6]).all():
        return 1


def drop_bomb_next_to_crate(self):
    t = self.transitions[-1]
    if 1 in t.state[0:4]:
        if t.action == "BOMB":
            return 1
    else:
        return 0


def in_danger_zone(self):
    t = self.transitions[-1]
    if t.state[6] == 1:
        return 1
    else:
        return 0
