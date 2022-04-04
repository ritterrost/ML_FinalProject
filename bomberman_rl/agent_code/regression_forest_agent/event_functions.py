"""
  Input: self, events, old_game_state, new_game_state
  Output: Amount of reward: int
"""

import numpy as np
import events as e
from .feature_functions import state_to_features_bfs_2 as state_to_features


WALKED_TO_COIN = "WALKED_TO_COIN"
WALKED_TO_CRATE = "WALKED_TO_CRATE"
WALKED_FROM_DANGER = "WALKED_FROM_DANGER"
STAYS_IN_DANGER_ZONE = "STAYS_IN_DANGER_ZONE"
HAS_NO_ESCAPE = "HAS_NO_ESCAPE"
BOMB_HAS_TARGETS = "BOMB_HAS_TARGETS"
BOMB_HAS_NOTHING = "BOMB_HAS_NOTHING"

A_TO_COORD = {
    "UP": [0, -1],
    "LEFT": [-1, 0],
    "RIGHT": [1, 0],
    "DOWN": [0, 1],
    "WAIT": [0, 0],
    "BOMB": [-1, -1],
}

BR = 3

def walked_towards_closest_coin(state, action):
    if (np.array(A_TO_COORD[action]) == state[0:2]).all(): # align action with direction of coin
        return 1


def walked_towards_closest_crate(state, action):
    if (np.array(A_TO_COORD[action]) == state[3:5]).all() and not state[5]==1: # align action with direction of coin
        return 1


def walked_from_danger(state, action):
    if state[8] == 0:  # check if in danger (distance to free tile)
        return 0
    else:
        if (np.array(A_TO_COORD[action]) == state[6:8]).all():  # align action with direction of free field
            return 1
        else:
            return -1


def bomb_target(old_game_state, events):
    if e.BOMB_DROPPED in events:
        others = np.asarray([xy for (n, s, b, xy) in old_game_state["others"]])
        b = old_game_state['self'][3]
        arena = old_game_state['field']
        
        if others.shape[0] != 0:
            for other in others:
                arena[other[0], other[1]] = 1

        bomb_targets = []
        u_b, r_b, d_b, l_b = False, False, False, False
        for i in range(BR + 1):
            if not u_b:
                if (arena[b[0] - i, b[1]] != -1):
                    bomb_targets.append(arena[b[0] - i, b[1]])
                else:
                    u_b = True
            if not d_b:
                if arena[b[0] + i, b[1]] != -1:
                    bomb_targets.append(arena[b[0] + i, b[1]])
                else:
                    d_b = True
            if not l_b:
                if arena[b[0], b[1] - i] != -1:
                    bomb_targets.append(arena[b[0], b[1] - i])
                else:
                    l_b = True
            if not r_b:
                if arena[b[0], b[1] + i] != -1:
                    bomb_targets.append(arena[b[0], b[1] + i])
                else:
                    r_b = True
        if 1 in bomb_targets:
            return 1
        else: 
            return -1
    else:
        return 0


def drop_bomb_next_to_crate(state, events):
    if state[5] == 1:  # distance to crate
        if e.BOMB_DROPPED in events:
            return 1


def has_no_escape(new_state, action):
    if action == 'BOMB':
        if new_state[8] > 4 or new_state[8] == -1:
            return 1


def reward_from_events(self, events: list[str]):
    game_rewards = {
        e.COIN_COLLECTED: 60,
        e.KILLED_SELF: -100,
        e.SURVIVED_ROUND: 20,
        e.GOT_KILLED: -150,
        BOMB_HAS_TARGETS: 15,
        BOMB_HAS_NOTHING: -40,
        # e.BOMB_DROPPED: -40,
        WALKED_FROM_DANGER: 30,
        STAYS_IN_DANGER_ZONE: -50,
        WALKED_TO_COIN: 15,
        WALKED_TO_CRATE: 5,
        HAS_NO_ESCAPE: -100,
        e.KILLED_OPPONENT: 300,
        e.INVALID_ACTION: -10,
    }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    # self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


def make_events(self, old_game_state, new_game_state, self_action, events):
    old_feat = state_to_features(old_game_state)
    new_feat = state_to_features(new_game_state)
    if walked_towards_closest_coin(old_feat, self_action):
        events.append(WALKED_TO_COIN)
    if walked_towards_closest_crate(old_feat, self_action):
        events.append(WALKED_TO_CRATE)
    walk_danger = walked_from_danger(old_feat, self_action)
    if walk_danger == 1:
        events.append(WALKED_FROM_DANGER)
    elif walk_danger == -1:
        events.append(STAYS_IN_DANGER_ZONE)
    drop_bomb = bomb_target(old_game_state, events)
    if drop_bomb == 1:
        events.append(BOMB_HAS_TARGETS)
    elif drop_bomb == -1:
        events.append(BOMB_HAS_NOTHING)
    if has_no_escape(new_feat, self_action):
        events.append(HAS_NO_ESCAPE)
        
    reward = reward_from_events(self, events)
    
    return old_feat, new_feat, reward