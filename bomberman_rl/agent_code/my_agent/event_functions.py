"""
  Input: self, events, old_game_state, new_game_state
  Output: Amount of reward: int
"""

import numpy as np
import events as e
from agent_code.my_agent.feature_functions import bfs_cc

BR = 3

# Events
MADE_SUGGESTED_MOVE = "MADE_SUGGESTED_MOVE"
IN_DANGER_ZONE = "IN_DANGER_ZONE"

WALKED_TOWARDS_CLOSEST_COIN = "WALKED_TOWARDS_CLOSEST_COIN"
WALKED_AWAY_FROM_CLOSEST_COIN = "WALKED_AWAY_FROM_CLOSEST_COIN"
COIN_LATERAL_MOVEMENT = "COIN_LATERAL_MOVEMENT"
TOUCH_CRATE = "TOUCH_CRATE"
WALKED_TOWARDS_CLOSEST_CRATE = "WALKED_TOWARDS_CLOSEST_CRATE"
WALKED_AWAY_FROM_CLOSEST_CRATE = "WALKED_AWAY_FROM_CLOSEST_CRATE"
CRATE_LATERAL_MOVEMENT = "CRATE_LATERAL_MOVEMENT"
DROP_BOMB_NEXT_TO_CRATE = "DROP_BOMB_NEXT_TO_CRATE"
WALKED_AWAY_FROM_BOMB = "WALKED_AWAY_FROM_BOMB"
WALKED_TOWARDS_BOMB = "WALKED_TOWARDS_BOMB"
WALKED_TOWARDS_FREE_TILE = "WALKED_TOWARDS_FREE_TILE"
STAYS_IN_DANGER_ZONE = "STAYS_IN_DANGER_ZONE"

A_TO_COORD = {
    "UP": [0, -1],
    "LEFT": [-1, 0],
    "RIGHT": [1, 0],
    "DOWN": [0, 1],
    "WAIT": [0, 0],
    "BOMB": [-1, -1],
}

def reward_from_events(self, events: list[str]):
    game_rewards = {
        ##Basic
        e.KILLED_SELF: -100,
        #e.GOT_KILLED: -100,
        #e.WAITED: -0.2,     #should be disabled when trying to learn how to place bombs?
        #e.INVALID_ACTION: -0.2,
        #e.SURVIVED_ROUND: 100,  #bad reward, we dont want passivity
        e.BOMB_DROPPED: -40,

        ##Coin collection
        e.COIN_COLLECTED: 60,

        ##Dodging bombs
        WALKED_TOWARDS_FREE_TILE: 40, #this leads to problems somehow
        STAYS_IN_DANGER_ZONE: -40,

        ##Destroying crates
        #e.CRATE_DESTROYED: 5,  #not a good reward since its not directly connected to last action
        DROP_BOMB_NEXT_TO_CRATE: 60,

        ##Killing enemies
        #e.KILLED_OPPONENT: 10

    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            print("event     :", event)
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

# def made_suggested_move(state, action):
#     if (np.array(A_TO_COORD[action]) == state[0:2]).all():
#         return 1


def drop_bomb_next_to_crate(feat, events):
    if feat[5] == 1: #feat[5] is distance to closest crate
        if "BOMB_DROPPED" in events:
            return 1
    else:
        return 0

#def in_danger_zone(feat):
#    if feat[8] == 0:
#       return 0
#    else:
#       return 1

def walked_towards_free_tile(old_state, new_state, old_feat):
    #check if in danger, i.e free tile is more than 0 away
    if old_feat[8] == 0:
        return 0
    else:
        dx_free, dy_free = old_feat[6], old_feat[7]
        _, _,  _, (x_old, y_old) = old_state["self"]
        _,_,_,(x_new,y_new) = new_state["self"]

        #print("dx, dy   :", x_new-x_old, y_new-y_old)
        #print("suggested:", dx_free, dy_free)

        if x_new-x_old == dx_free and y_new-y_old == dy_free:
            return 1
        else:
            return -1

#not used right now
def walked_towards_closest_coin(self, events, old_game_state, new_game_state):
    return 0

def walked_towards_closest_crate(self, events, old_game_state, new_game_state):
    return 0

def walked_towards_closest_player(self, events, old_game_state, new_game_state):
    return 0

def walked_away_from_bomb(self, events, old_game_state, new_game_state):
    return 0