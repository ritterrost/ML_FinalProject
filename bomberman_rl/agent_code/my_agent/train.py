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

# This is only an example!
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 1  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
COIN_K = 1
# ALPHA = 0.1
BATCH_SIZE = 2000
GAMMA = 0.1

# Events
MADE_SUGGESTED_MOVE = "MADE_SUGGESTED_MOVE"
DROP_BOMB_NEXT_TO_CRATE = "DROP_BOMB_NEXT_TO_CRATE"
IN_DANGER_ZONE = "IN_DANGER_ZONE"


# for convenience
A_NUM = 6


def response_func(self, gamma=GAMMA):
    transition = self.transitions[-1]
    next_q_value = callbacks.Q_func(self, feat=transition.next_state)
    return transition.reward + gamma * np.max(next_q_value)


def forest_update(self):
    # select random batch of transitions and updates all actions at once
    for idx in np.arange(A_NUM):
        # batch for action denoted by idx
        if len(self.feat_history[idx]) > 0:
            selection_mask = np.random.choice(
                np.arange(len(self.feat_history[idx])), size=BATCH_SIZE
            )
            X = np.array(self.feat_history[idx])[selection_mask]
            y = np.array(self.target_history[idx])[selection_mask]
            self.forests[idx].fit(X, y)


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
        if event_functions.made_suggested_move(feature_functions.state_to_features_bfs_2(old_game_state), self_action) == 1:
            events.append(MADE_SUGGESTED_MOVE)
        if event_functions.drop_bomb_next_to_crate(feature_functions.state_to_features_bfs_2(old_game_state), self_action) == 1:
            events.append(DROP_BOMB_NEXT_TO_CRATE)
        self.transitions.append(
            Transition(
                feature_functions.state_to_features_bfs_2(old_game_state),
                self_action,
                feature_functions.state_to_features_bfs_2(new_game_state),
                # reward_from_events(self, events),
                total_rewards(self, events, old_game_state, new_game_state),
            )
        )
        # if event_functions.in_danger_zone(self) == 1:
        #     events.append(IN_DANGER_ZONE)
        self.reward_data += total_rewards(self, events, old_game_state, new_game_state)

        feat = feature_functions.state_to_features_bfs_2(old_game_state)
        Y_tt = response_func(self)
        idx = callbacks.A_TO_NUM[self_action]

        self.target_history[idx].append(Y_tt)
        self.feat_history[idx].append(feat)

        self.logger.info(f"feature_vec: {feat}")
        # self.logger.info(f"self.target_history: {self.target_history}")
        # self.logger.info(f"self.feat_history: {self.feat_history}")


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    self.transitions.append(
        Transition(
            feature_functions.state_to_features_bfs_2(last_game_state),
            last_action,
            None,
            reward_from_events(self, events),
        )
    )
    with open('data.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow((last_game_state["round"], last_game_state["self"][1], last_game_state["step"], self.reward_data))

    # reset reward counter for plotting
    self.reward_data = 0

    # update forests for all actions
    forest_update(self)

    # Store the model
    # if self.round_counter == 1000:
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.forests, file)


def total_rewards(self, events, old_game_state, new_game_state):
    rewards = [reward_from_events(self, events)]

    # self.plotting.store_data(rewards)
    return sum(rewards)


def reward_from_events(self, events: List[str]):
    game_rewards = {
        e.COIN_COLLECTED: 5,
        # e.KILLED_SELF: -2,
        e.CRATE_DESTROYED: 1,
        e.KILLED_OPPONENT: 10,
        e.GOT_KILLED: -2,
        MADE_SUGGESTED_MOVE: 5,
        DROP_BOMB_NEXT_TO_CRATE: 1,
        # IN_DANGER_ZONE: -5,

    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
