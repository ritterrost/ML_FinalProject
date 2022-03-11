from collections import namedtuple, deque
import pickle
from subprocess import call
from typing import List
import numpy as np
import events as e

from agent_code.my_agent import event_functions
from agent_code.my_agent import feature_functions
from agent_code.my_agent import callbacks

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
WALKED_TOWARDS_CLOSEST_COIN = "WALKED_TOWARDS_CLOSEST_COIN"
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
    # to improve training speed
    # self.round_counter = 0
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)


def game_events_occurred(
    self,
    old_game_state: dict,
    self_action: str,
    new_game_state: dict,
    events: List[str],
):

    # custom events use event_functions here
    if old_game_state is not None:
        self.transitions.append(
            Transition(
                feature_functions.state_to_features_bfs_2(old_game_state),
                self_action,
                feature_functions.state_to_features_bfs_2(new_game_state),
                # reward_from_events(self, events),
                total_rewards(self, events, old_game_state, new_game_state),
            )
        )
        if event_functions.walked_towards_closest_coin(self) == 1:
            events.append(WALKED_TOWARDS_CLOSEST_COIN)
        if event_functions.drop_bomb_next_to_crate(self) == 1:
            events.append(DROP_BOMB_NEXT_TO_CRATE)
        if event_functions.in_danger_zone(self) == 1:
            events.append(IN_DANGER_ZONE)

        feat = feature_functions.state_to_features_bfs_2(old_game_state)
        Y_tt = response_func(self)
        idx = callbacks.A_TO_NUM[self_action]

        self.target_history[idx].append(Y_tt)
        self.feat_history[idx].append(feat)

        self.logger.info(f"feature_vec: {feat}")
        # self.logger.info(f"self.target_history: {self.target_history}")
        # self.logger.info(f"self.feat_history: {self.feat_history}")


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    # self.round_counter += 1
    self.transitions.append(
        Transition(
            feature_functions.state_to_features_bfs_2(last_game_state),
            last_action,
            None,
            reward_from_events(self, events),
        )
    )

    # update forests for all actions
    forest_update(self)

    # Store the model
    # if self.round_counter == 1000:
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.forests, file)


def total_rewards(self, events, old_game_state, new_game_state):
    rewards = [reward_from_events(self, events)]
    return sum(rewards)


def reward_from_events(self, events: List[str]):
    game_rewards = {
        e.COIN_COLLECTED: 5,
        e.KILLED_SELF: -2,
        e.CRATE_DESTROYED: 2,
        e.KILLED_OPPONENT: 10,
        e.GOT_KILLED: -2,
        WALKED_TOWARDS_CLOSEST_COIN: 1,
        DROP_BOMB_NEXT_TO_CRATE: 5,
        IN_DANGER_ZONE: -5,

    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    # self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
