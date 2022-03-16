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
        e.KILLED_SELF: -50,
        e.GOT_KILLED: -50,
        #e.WAITED: -0.2,     #should be disabled when trying to learn how to place bombs?
        #e.INVALID_ACTION: -0.2,
        #e.SURVIVED_ROUND: 100,
        e.BOMB_DROPPED: -5,

        ##Coin collection
        e.COIN_COLLECTED: 20,
        #WALKED_TOWARDS_CLOSEST_COIN: 0.1,
        #WALKED_AWAY_FROM_CLOSEST_COIN: -0.1,
        #COIN_LATERAL_MOVEMENT: 0,

        ##Dodging bombs
        #IN_DANGER_ZONE: -5,
        #WALKED_AWAY_FROM_BOMB: 3,
        #WALKED_TOWARDS_BOMB: -3,
        #WALKED_TOWARDS_FREE_TILE: 10,  #not necessary for STAYS_IN_DANGER_ZONE 1/5 of dying and COIN_COLLECTED 2/5
        STAYS_IN_DANGER_ZONE: -10,

        ##Destroying crates
        #e.CRATE_DESTROYED: 5,  #not a good reward since its not directly connected to last action
        #WALKED_TOWARDS_CLOSEST_CRATE: 0.02,
        #WALKED_AWAY_FROM_CLOSEST_CRATE: -0.02,
        #CRATE_LATERAL_MOVEMENT:-0.02,
        DROP_BOMB_NEXT_TO_CRATE: 15,
        #TOUCH_CRATE: 5,

        ##Killing enemies
        e.KILLED_OPPONENT: 10

    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

# def made_suggested_move(state, action):
#     if (np.array(A_TO_COORD[action]) == state[0:2]).all():
#         return 1


def drop_bomb_next_to_crate(state, action):
    if state[5] == 1: #state[5] is distance to closest crate
        if action == "BOMB":
            return 1
    else:
        return 0

#def in_danger_zone(state):
#    if state[12] == 1:
#       return 1
#    else:
#       return 0

def walked_towards_free_tile(state, new_state, action):
    arena = state["field"]
    others = np.asarray([xy for (n, s, b, xy) in state["others"]])
    coins = np.asarray(state["coins"])
    bombs = state["bombs"]
    bombs_xy = np.asarray([[bomb_x, bomb_y] for ((bomb_x, bomb_y), _) in bombs])
    explosion_map = state["explosion_map"]
    _, _,  bombs_left, (x, y) = state["self"]
    pos_self = np.asarray((x, y))

    _,_,_,(x_new,y_new) = new_state["self"]
    pos_diff = np.asarray((x_new,y_new)) - pos_self

    #look for free tile
    orig_arena = arena
    is_in_danger_zone = False
    if bombs_xy.shape[0] != 0:
        for b in bombs_xy:
            u_b, r_b, d_b, l_b = False, False, False, False
            for i in range(BR + 1):
                if not u_b:
                    if arena[b[0] - i, b[1]] != -1:
                        arena[b[0] - i, b[1]] = 4
                    else:
                        u_b = True
                if not d_b:
                    if arena[b[0] + i, b[1]] != -1:
                        arena[b[0] + i, b[1]] = 4
                    else:
                        d_b = True
                if not l_b:
                    if arena[b[0], b[1] - i] != -1:
                        arena[b[0], b[1] - i] = 4
                    else:
                        l_b = True
                if not r_b:
                    if arena[b[0], b[1] + i] != -1:
                        arena[b[0], b[1] + i] = 4
                    else:
                        r_b = True

    if arena[pos_self[0], pos_self[1]] == 4:
        is_in_danger_zone = True
    
    if not is_in_danger_zone:
        return 0 #if not in danger disregard free step

    else:
        explosion_array = np.argwhere(explosion_map == 1)
        if explosion_array.shape[0] != 0:
            arena[explosion_array[:, 0], explosion_array[:, 1]] = 5

        free_step = np.array([1, 1])
        free_dist = -1
        next_coord, free_dist = bfs_cc(orig_arena, arena, pos_self, "free")
        if next_coord is not None:
            free_step = next_coord - pos_self
        else:
            return 0 #if no free step available do not punish

        #check if moved in direction of "free step", note that the board is showed transposed, i.e. shows (y,x)
        if np.all(free_step == pos_diff):
            return 1
        else:
            return 0.5



#not used right now
def walked_towards_closest_coin(self, events, old_game_state, new_game_state):
    return 0

def walked_towards_closest_crate(self, events, old_game_state, new_game_state):
    return 0

def walked_towards_closest_player(self, events, old_game_state, new_game_state):
    return 0

def walked_away_from_bomb(self, events, old_game_state, new_game_state):
    return 0