from collections import namedtuple, deque
import pickle
from subprocess import call
from typing import List
import numpy as np
from agent_code.my_agent.plots import Plotting
import events as e

from agent_code.my_agent import event_functions
from agent_code.my_agent import feature_functions
from agent_code.my_agent import callbacks

# for plotting
import csv
import pandas as pd
import numpy as np

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

# This is only an example!
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 1  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
COIN_K = 1
# ALPHA = 0.1
BATCH_SIZE = 2000
GAMMA = 0.7 #0.5 seems good


# for convenience
A_NUM = 6
A_IDX = np.arange(0, A_NUM, 1, dtype="int")

def forest_update(self):
    #select random batch of transitions and updates all actions at once

    for idx in A_IDX:
        if len(self.feat_history[idx])>0:
            X = np.array(self.feat_history[idx])
            next_q_value = np.array([self.forests[i].predict(np.array(self.next_feat_history[idx])) for i in A_IDX])
            y = np.array(self.reward_history[idx]) + GAMMA*np.max(next_q_value, axis = 0)
            self.forests[idx].fit(np.array(X),np.array(y))
        else:
            continue

def setup_training(self):
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.reward_data = 0 # for plotting

    with open('data.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['round', 'score', 'survival time', 'total rewards'])


def game_events_occurred(
    self,
    old_game_state: dict,
    self_action: str,
    new_game_state: dict,
    events: List[str],
):

    # custom events use event_functions here
    if old_game_state is not None:
        old_feat = feature_functions.state_to_features_bfs_2(old_game_state)
        #if event_functions.made_suggested_move(feature_functions.state_to_features_bfs_2(old_game_state), self_action) == 1:
        #    events.append(MADE_SUGGESTED_MOVE)

        if event_functions.drop_bomb_next_to_crate(old_feat, self_action) == 1:  #DONE
            events.append(DROP_BOMB_NEXT_TO_CRATE)

        ##if event_functions.in_danger_zone(old_feat) == 1:
        ##    events.append(IN_DANGER_ZONE)

        free_tile_event = event_functions.walked_towards_free_tile(old_game_state, new_game_state, self_action)
        if free_tile_event == 1:
            events.append(WALKED_TOWARDS_FREE_TILE)
        elif free_tile_event == 0.5:
            events.append(STAYS_IN_DANGER_ZONE)


        self.transitions.append(
            Transition(
                feature_functions.state_to_features_bfs_2(old_game_state),
                self_action,
                feature_functions.state_to_features_bfs_2(new_game_state),
                # reward_from_events(self, events),
                total_rewards(self, events, old_game_state, new_game_state),
            )
        )

        feat = feature_functions.state_to_features_bfs_2(old_game_state)
        idx = callbacks.A_TO_NUM[self_action]

        self.feat_history[idx].append(feat)
        self.reward_history[idx].append(total_rewards(self, events, old_game_state, new_game_state))
        self.reward_data += self.reward_history[idx][-1]
        self.next_feat_history[idx].append(feature_functions.state_to_features_bfs_2(new_game_state))

        #self.logger.info(f"feature_vec: {feat}")
        # self.logger.info(f"self.target_history: {self.target_history}")
        # self.logger.info(f"self.feat_history: {self.feat_history}")


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    self.transitions.append(
        Transition(
            feature_functions.state_to_features_bfs_2(last_game_state),
            last_action,
            None,
            event_functions.reward_from_events(self, events),
        )
    )
    with open('data.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow((last_game_state["round"], last_game_state["self"][1], last_game_state["step"], self.reward_data))

    # reset reward counter for plotting
    self.reward_data = 0

    #update forests of all actions
    forest_update(self)

    # Store the model
    with open("current_model/my-saved-model.pt", "wb") as file:
        pickle.dump(self.forests, file)
    with open("current_model/feature_history.pt", "wb") as file:
        pickle.dump(self.feat_history, file)
    with open("current_model/next_feature_history.pt", "wb") as file:
        pickle.dump(self.next_feat_history, file)
    with open("current_model/reward_history.pt", "wb") as file:
        pickle.dump(self.reward_history, file)


def total_rewards(self, events, old_game_state, new_game_state):
    rewards = [event_functions.reward_from_events(self, events)]

    # self.plotting.store_data(rewards)
    return sum(rewards)