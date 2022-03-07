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
TRANSITION_HISTORY_SIZE = 1  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
COIN_K = 1
#ALPHA = 0.1
BATCH_SIZE = 10000
L = 1
SAMPLE_SIZE = 10
GAMMA = 0.1

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"
WALKED_TOWARDS_CLOSEST_COIN = "WALKED_TOWARDS_CLOSEST_COIN"
WALKED_AWAY_FROM_CLOSEST_COIN = "WALKED_AWAY_FROM_CLOSEST_COIN"
LATERAL_MOVEMENT = "LATERAL_MOVEMENT"

#for convenience
A_IDX = np.arange(0,6,1,dtype='int')
FEAT_DIM = 6

def TD_target_respones(self, gamma = GAMMA):
    transition = self.transitions[-1]
    # Q function of next value
    next_q_value = Q_func(self, feat=transition.next_state)

    return transition.reward + gamma * np.max(next_q_value)


def forest_update(self):
    #select random batch of transitions and updates all actions at once

    for idx in A_IDX:
        #batch for action dneoted by idx
        if len(self.feat_history[idx])>0:
            selection_mask = np.random.choice(np.arange(0,len(self.feat_history[idx]), dtype='int'), size = BATCH_SIZE)
            X = np.array(self.feat_history[idx])[selection_mask]
            y = np.array(self.target_history[idx])[selection_mask]
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
        if walked_towards_closest_coin(self, events, old_game_state, new_game_state) == 1:
            events.append(WALKED_TOWARDS_CLOSEST_COIN)
        if walked_towards_closest_coin(self, events, old_game_state, new_game_state) == -1:
            events.append(WALKED_AWAY_FROM_CLOSEST_COIN)
        if walked_towards_closest_coin(self, events, old_game_state, new_game_state) == 0.5:
            events.append(LATERAL_MOVEMENT)

    # state_to_features is defined in callbacks.py
    if old_game_state is not None:
        self.transitions.append(
            Transition(
                state_to_features(old_game_state),
                self_action,
                state_to_features(new_game_state),
                #reward_from_events(self, events),
                total_rewards(self, events, old_game_state, new_game_state)
            )
        )
        feat = state_to_features(old_game_state)

        Y_tt = TD_target_respones(self)
        idx = A_TO_NUM[self_action]

        self.target_history[idx].append(Y_tt)
        self.feat_history[idx].append(feat)
        #self.logger.info(f"self.target_history: {self.target_history}")
        #self.logger.info(f"self.feat_history: {self.feat_history}")


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    #self.logger.debug(
    #    f'Encountered event(s) {", ".join(map(repr, events))} in final step'
    #)
    self.transitions.append(
        Transition(
            state_to_features(last_game_state),
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

def total_rewards(self, events, old_game_state, new_game_state):
    rewards = [reward_from_events(self, events)]
    return sum(rewards)

def reward_from_events(self, events: List[str]):
    game_rewards = {
        e.COIN_COLLECTED: 10,
        WALKED_TOWARDS_CLOSEST_COIN: 1,
        WALKED_AWAY_FROM_CLOSEST_COIN: -1,
        LATERAL_MOVEMENT: 0,
        #e.WAITED: -2,
        #e.INVALID_ACTION: -5,
        e.KILLED_SELF: -100,
        #e.SURVIVED_ROUND: 100,
        e.CRATE_DESTROYED: 2,
        #e.BOMB_DROPPED: 2
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
    wall_above = old_arena[old_pos[0], old_pos[1]+1] == -1
    wall_below = old_arena[old_pos[0], old_pos[1]-1] == -1
    wall_right = old_arena[old_pos[0]+1, old_pos[1]] == -1
    wall_left = old_arena[old_pos[0]-1, old_pos[1]] == -1

    #position of coins
    old_coins, new_coins = old_game_state["coins"], new_game_state["coins"]
    if len(old_coins)==0:
        return 0
    elif len(new_coins)==0:
        return 0
    else:
        old_coin_xy, new_coin_xy = np.array([[coin_x,coin_y] for (coin_x,coin_y) in old_coins]), np.array([[coin_x,coin_y] for (coin_x,coin_y) in new_coins])
        #old closest coin
        old_closest_index = np.argsort(np.linalg.norm(old_coin_xy-old_pos, ord=L, axis=1))[0]
        coin_pos = old_coin_xy[old_closest_index]
        #direction of coin relative to old position
        coin_direction = coin_pos - old_pos
        #exclude case where player has to move laterally
        lateral_movement_necessary = (np.all(coin_direction == [0,1]) and wall_above) or (np.all(coin_direction == [0,-1]) and wall_below)\
                                   or (np.all(coin_direction == [1,0]) and wall_right) or (np.all(coin_direction == [-1,0]) and wall_left)
        if lateral_movement_necessary: return 0.5
        #check if there still is a coin at position of old closest coin
        if e.COIN_COLLECTED in events:
            return 1
        if coin_pos in new_coin_xy:
            #give reward if walked in direction, punish if walked away
            diff = np.linalg.norm(coin_pos-new_pos, ord=L)-np.linalg.norm(coin_pos-old_pos, ord=L)
            return np.sign(-diff) #+/-1 after 1 step
        else:
            return 0