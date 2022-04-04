"""
  Input: self, events, old_game_state, new_game_state
  Output: Amount of reward: int
"""

import numpy as np
import events as e

DROP_BOMB_CLOSE_TO_OTHER = "DROP_BOMB_CLOSE_TO_OTHER"
WALKED_TO_COIN = "WALKED_TO_COIN"
WALKED_TO_CRATE = "WALKED_TO_CRATE"
DROP_BOMB_NEXT_TO_CRATE = "DROP_BOMB_NEXT_TO_CRATE"
WALKED_FROM_DANGER = "WALKED_FROM_DANGER"
STAYS_IN_DANGER_ZONE = "STAYS_IN_DANGER_ZONE"
HAS_NO_ESCAPE = "HAS_NO_ESCAPE"


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
    if state[5] == 0:  # check if in danger (distance to free tile)
        return 0
    else:
        if (np.array(A_TO_COORD[action]) == state[3:5]).all():  # align action with direction of free field
            return 1
        else:
            return -1


def drop_bomb_next_to_crate(state, events):
    if state[8] == 1:  # distance to crate
        if e.BOMB_DROPPED in events:
            return 1


def has_no_escape(new_state, action):
    if action == 'BOMB':
        if new_state[5] > 4 or new_state[5] == -1:
            return 1

def drop_bomb_close_to_other(state, action):
    if state[11] == 1:
        if action == "BOMB":
            return 1

def walked_towards_closest_crate(state, action):
    if (np.array(A_TO_COORD[action]) == state[6:8]).all() and not state[8]==1: # align action with direction of coin
        return 1


def reward_from_events(self, events: list[str]):
    game_rewards = {
        # DROP_BOMB_NEXT_TO_CRATE: 40,
        e.COIN_COLLECTED: 60,
        e.KILLED_SELF: -100,
        e.SURVIVED_ROUND: 100,
        # e.KILLED_OPPONENT: 0,
        # e.GOT_KILLED: -100,
        DROP_BOMB_NEXT_TO_CRATE: 60,
        e.BOMB_DROPPED: -20,
        WALKED_FROM_DANGER: 40,
        STAYS_IN_DANGER_ZONE: -40,
        WALKED_TO_COIN: 10,
        HAS_NO_ESCAPE: -100,
        # e.KILLED_OPPONENT: 100,
        e.INVALID_ACTION: -5,

    }
        
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum