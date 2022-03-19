from agent_code.my_agent.callbacks import A_TO_NUM, Q_func
from collections import namedtuple, deque
import pickle
from typing import List
import events as e
from agent_code.my_agent.callbacks import state_to_features
import numpy as np

# This is only an example!
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 2  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
COIN_K = 1
#ALPHA = 0.1
BATCH_SIZE = 1000000
L = 1
SAMPLE_SIZE = 10
GAMMA = 0.1

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"
WALKED_TOWARDS_CLOSEST_COIN = "WALKED_TOWARDS_CLOSEST_COIN"
WALKED_AWAY_FROM_CLOSEST_COIN = "WALKED_AWAY_FROM_CLOSEST_COIN"
COIN_LATERAL_MOVEMENT = "COIN_LATERAL_MOVEMENT"
WALKED_TOWARDS_CLOSEST_CRATE = "WALKED_TOWARDS_CLOSEST_COIN"
WALKED_AWAY_FROM_CLOSEST_CRATE = "WALKED_AWAY_FROM_CLOSEST_COIN"
CRATE_LATERAL_MOVEMENT = "COIN_LATERAL_MOVEMENT"
BOMB_NEXT_TO_CRATE = "BOMB_NEXT_TO_CRATE"
WALKED_AWAY_FROM_BOMB = "WALKED_AWAY_FROM_BOMB"
WALKED_TOWARDS_BOMB = "WALKED_TOWARDS_BOMB"

#for convenience
A_IDX = np.arange(0,6,1,dtype='int')
FEAT_DIM = 6

#def TD_target_respones(self, gamma = GAMMA):
#    transition = self.transitions[-1]
#    # Q function of next value
#    next_q_value = Q_func(self, feat=transition.next_state)
#
#    return transition.reward + gamma * np.max(next_q_value)


def forest_update(self):
    #select random batch of transitions and updates all actions at once
    ##########WE SHOULD USE LAST TREE FOR TARGET PREDICTION NOT SAVED VALUES!!!

    for idx in A_IDX:
        #batch for action denoted by idx
        if len(self.feat_history[idx])>0:
            #selection_mask = np.random.choice(np.arange(0,len(self.feat_history[idx]), dtype='int'), size = BATCH_SIZE)
            X = np.array(self.feat_history[idx])#[selection_mask]
            #y = np.array(self.target_history[idx])#[selection_mask]
            next_q_value = np.array([self.forests[i].predict(np.array(self.next_feat_history[idx])) for i in A_IDX])
            y = np.array(self.reward_history[idx]) + GAMMA*np.max(next_q_value, axis = 0)
            self.forests[idx].fit(np.array(X),np.array(y))
        else:
            continue


def setup_training(self):
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)


def game_events_occurred(
    self,
    old_game_state: dict,
    self_action: str,
    new_game_state: dict,
    events: List[str],
    ):

    # custom events
    if old_game_state is not None:
        w_coin = walked_towards_closest_coin(self, events, old_game_state, new_game_state)
        if w_coin == 1:
            events.append(WALKED_TOWARDS_CLOSEST_COIN)
        if w_coin == -1:
            events.append(WALKED_AWAY_FROM_CLOSEST_COIN)
        if w_coin == 0.5:
            events.append(COIN_LATERAL_MOVEMENT)
        w_crate = walked_towards_closest_crate(self, events, old_game_state, new_game_state)
        if w_crate == 1:
            events.append(WALKED_TOWARDS_CLOSEST_CRATE)
        if w_crate == -1:
            events.append(WALKED_AWAY_FROM_CLOSEST_CRATE)
        if w_crate == 0.5:
            events.append(CRATE_LATERAL_MOVEMENT)
        if dropped_bomb_next_to_crate(self, events, old_game_state, new_game_state) == 1:
            events.append(BOMB_NEXT_TO_CRATE)
        w_bomb = walked_away_from_bomb(self, events, old_game_state, new_game_state)
        if w_bomb == 1 or w_bomb == 0.5:
            events.append(WALKED_AWAY_FROM_BOMB)
        if w_bomb == -1 or w_bomb == -0.5:
            events.append(WALKED_TOWARDS_BOMB)


    # state_to_features is defined in callbacks.py
    if old_game_state is not None:
        self.transitions.append(
            Transition(
                state_to_features(self, old_game_state),
                self_action,
                state_to_features(self, new_game_state),
                #reward_from_events(self, events),
                total_rewards(self, events, old_game_state, new_game_state)
            )
        )

        feat = state_to_features(self, old_game_state)
        #Y_tt = TD_target_respones(self)
        #self.target_history[idx].append(Y_tt)
        idx = A_TO_NUM[self_action]

        self.feat_history[idx].append(feat)
        self.reward_history[idx].append(total_rewards(self, events, old_game_state, new_game_state)) #MUST BE BEFORE NEXT_FEATURE IS CALCULATED!
        self.next_feat_history[idx].append(state_to_features(self, new_game_state))
        #self.logger.info(f"self.target_history: {self.target_history}")
        #self.logger.info(f"self.feat_history: {self.feat_history}")


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    #self.logger.debug(
    #    f'Encountered event(s) {", ".join(map(repr, events))} in final step'
    #)
    self.transitions.append(
        Transition(
            state_to_features(self, last_game_state),
            last_action,
            None,
            reward_from_events(self, events),
        )
    )

    #update forests of all actions
    forest_update(self)

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.forests, file)
    with open("feature_history.pt", "wb") as file:
        pickle.dump(self.feat_history, file)
    with open("next_feature_history.pt", "wb") as file:
        pickle.dump(self.next_feat_history, file)
    with open("reward_history.pt", "wb") as file:
        pickle.dump(self.reward_history, file)

def total_rewards(self, events, old_game_state, new_game_state):
    rewards = [reward_from_events(self, events)]
    return sum(rewards)

def reward_from_events(self, events: List[str]):
    game_rewards = {
        e.COIN_COLLECTED: 1,
        WALKED_TOWARDS_CLOSEST_COIN: 0.1,
        WALKED_AWAY_FROM_CLOSEST_COIN: -0.1,
        COIN_LATERAL_MOVEMENT: 0,
        #WALKED_TOWARDS_CLOSEST_CRATE: 0.02,
        #WALKED_AWAY_FROM_CLOSEST_CRATE: -0.02,
        #CRATE_LATERAL_MOVEMENT:-0.02,
        #BOMB_NEXT_TO_CRATE: 15,
        #e.CRATE_DESTROYED: 5,
        WALKED_AWAY_FROM_BOMB: 3,
        WALKED_TOWARDS_BOMB: -3,
        e.WAITED: -0.2,     #should be disabled when trying to learn how to place bombs
        e.INVALID_ACTION: -0.2,
        e.KILLED_SELF: -10,
        #e.SURVIVED_ROUND: 100,
        #e.CRATE_DESTROYED: 0,
        e.BOMB_DROPPED: -5
        # e.KILLED_OPPONENT: 2,
        # PLACEHOLDER_EVENT: -0.1,  # idea: the custom event is bad
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    #self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

#for custom events
def walked_towards_closest_coin(self, events, old_game_state, new_game_state):
    old_pos, new_pos = old_game_state["self"][-1], new_game_state["self"][-1]
    #field
    old_arena = old_game_state["field"]
    #surrounding walls
    x,y = old_pos
    explosion_map = old_game_state["explosion_map"]
    wall_above = old_arena[x, y+1] == -1 or explosion_map[x, y+1] != 0
    wall_below = old_arena[x, y-1] == -1 or explosion_map[x, y-1] != 0
    wall_right = old_arena[x+1, y] == -1 or explosion_map[x+1, y] != 0
    wall_left = old_arena[x-1, y] == -1 or explosion_map[x-1, y] != 0

    #position of coins
    old_coins, new_coins = old_game_state["coins"], new_game_state["coins"]
    if len(old_coins)==0 or len(new_coins)==0:
        return 0
    else:
        new_coin_xy = np.array([[coin_x,coin_y] for (coin_x,coin_y) in new_coins])
        #direction of coin relative to old position
        direction = self.old_closest_coin - old_pos
        #exclude case where player has to move laterally
        lateral_movement_necessary = (np.all(direction == [0,1]) and wall_above) or (np.all(direction == [0,-1]) and wall_below)\
                                   or (np.all(direction == [1,0]) and wall_right) or (np.all(direction == [-1,0]) and wall_left)
        if lateral_movement_necessary: return 0.5
        #check if there still is a coin at position of old closest coin
        if e.COIN_COLLECTED in events:
            return 1
        if self.old_closest_coin in new_coin_xy:
            #give reward if walked in direction, punish if walked away
            diff = np.linalg.norm(self.old_closest_coin-new_pos, ord=L)-np.linalg.norm(self.old_closest_coin-old_pos, ord=L)
            return np.sign(-diff) #+/-1 after 1 step
        else:
            return 0

def walked_towards_closest_crate(self, events, old_game_state, new_game_state):
    old_pos, new_pos = old_game_state["self"][-1], new_game_state["self"][-1]
    #field
    old_arena, new_arena = old_game_state["field"], new_game_state["field"]
    #surrounding walls
    x,y = old_pos
    explosion_map = old_game_state["explosion_map"]
    wall_above = old_arena[x, y+1] == -1 or explosion_map[x, y+1] != 0
    wall_below = old_arena[x, y-1] == -1 or explosion_map[x, y-1] != 0
    wall_right = old_arena[x+1, y] == -1 or explosion_map[x+1, y] != 0
    wall_left = old_arena[x-1, y] == -1 or explosion_map[x-1, y] != 0
    
    #position of barrel
    old_barrels, new_barrels = np.argwhere(old_arena>0), np.argwhere(new_arena>0)
    if len(old_barrels)==0 or len(new_barrels) ==0:
        return 0
    else:
        #direction of coin relative to old position
        direction = self.old_closest_barrel - old_pos
        #exclude case where player has to move laterally
        lateral_movement_necessary = (np.all(direction == [0,1]) and wall_above) or (np.all(direction == [0,-1]) and wall_below)\
                                   or (np.all(direction == [1,0]) and wall_right) or (np.all(direction == [-1,0]) and wall_left)
        if lateral_movement_necessary: return 0.5
        #check if there still is a coin at position of old closest coin
        if e.CRATE_DESTROYED in events:
            return 1
        if self.old_closest_barrel in new_barrels:
            #give reward if walked in direction, punish if walked away
            diff = np.linalg.norm(self.old_closest_barrel-new_pos, ord=L)-np.linalg.norm(self.old_closest_barrel-old_pos, ord=L)
            return np.sign(-diff) #+/-1 after 1 step
        else:
            return 0

def walked_towards_closest_player(self, events, old_game_state, new_game_state):
    old_pos, new_pos = old_game_state["self"][-1], new_game_state["self"][-1]
    #field
    old_arena, new_arena = old_game_state["field"], new_game_state["field"]
    #surrounding walls
    x,y = old_pos
    explosion_map = old_game_state["explosion_map"]
    wall_above = old_arena[x, y+1] == -1 or explosion_map[x, y+1] != 0
    wall_below = old_arena[x, y-1] == -1 or explosion_map[x, y-1] != 0
    wall_right = old_arena[x+1, y] == -1 or explosion_map[x+1, y] != 0
    wall_left = old_arena[x-1, y] == -1 or explosion_map[x-1, y] != 0
    
    #position of enemys
    old_others = np.array([xy for (n, s, b, xy) in old_game_state["others"]])
    new_others = np.array([xy for (n, s, b, xy) in new_game_state["others"]])
    if len(old_others)==0 or len(new_others) ==0:
        return 0
    else:
        #direction of coin relative to old position
        direction = self.old_closest_player - old_pos
        #exclude case where player has to move laterally
        lateral_movement_necessary = (np.all(direction == [0,1]) and wall_above) or (np.all(direction == [0,-1]) and wall_below)\
                                   or (np.all(direction == [1,0]) and wall_right) or (np.all(direction == [-1,0]) and wall_left)
        if lateral_movement_necessary: return 0.5
        #give reward if walked in direction, punish if walked away
        diff = np.linalg.norm(self.old_closest_player-new_pos, ord=L)-np.linalg.norm(self.old_closest_player-old_pos, ord=L)
        return np.sign(-diff) #+/-1 after 1 step

def dropped_bomb_next_to_crate(self, events, old_game_state, new_game_state):
    closest_crate = self.old_closest_barrel
    if e.BOMB_DROPPED in events:
        bomb_xy = [[bomb_x, bomb_y] for ((bomb_x, bomb_y), bomb_t) in new_game_state["bombs"]]
        distance = closest_crate-np.array(bomb_xy)
        if np.any(np.linalg.norm(distance, ord=1)==1):
            return 1
    else: return 0

def walked_away_from_bomb(self, events, old_game_state, new_game_state):
    old_pos, new_pos = np.array(old_game_state["self"][-1]), np.array(new_game_state["self"][-1])
    #field
    old_arena, new_arena = old_game_state["field"], new_game_state["field"]
    bomb_xys = np.array([[bomb_x, bomb_y] for ((bomb_x, bomb_y), bomb_t) in old_game_state["bombs"]])
    num_bombs=len(bomb_xys)
    if num_bombs==0:
        return 0
    bomb_ranges = []
    for bomb in bomb_xys:
        x,y = bomb
        bomb_ranges.append([[x,y], [x,y+1], [x,y+2], [x,y+3], [x,y+4], [x,y-1], [x,y-2], [x,y-3], [x,y-4],\
                                   [x+1,y], [x+2,y], [x+3,y], [x+4,y], [x-1,y], [x-2,y], [x-3,y], [x-4,y]])
    bomb_ranges_r = np.array(bomb_ranges).reshape(num_bombs*17,2)
    if old_pos in bomb_ranges_r and not new_pos in bomb_ranges_r:
        return 1
    elif new_pos in bomb_ranges_r and not old_pos in bomb_ranges_r:
        return -1
    elif old_pos in bomb_ranges_r and new_pos in bomb_ranges_r:
        if num_bombs == 1:
            bomb_pos = bomb_xys
        else:
            idx = np.argwhere(np.all(bomb_ranges == old_pos, axis=-1))
            bomb_pos = bomb_xys[idx]
        #give reward if player walks away from dangerous bomb
        diff = np.linalg.norm(bomb_pos-new_pos, ord=L)-np.linalg.norm(bomb_pos-old_pos, ord=L)
        if diff>0:
            return 0.5
        else:
            return -0.5

    else: return 0