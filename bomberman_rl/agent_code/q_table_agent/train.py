from os import stat
import pickle
from typing import List
import numpy as np

from .event_functions import (
    walked_towards_closest_coin,
    walked_from_danger,
    drop_bomb_next_to_crate,
    has_no_escape,
    reward_from_events,
)
from .feature_functions import state_to_features
from .callbacks import A_TO_NUM, Q_func, ACTIONS

import csv

# Events
WALKED_TO_COIN = "WALKED_TO_COIN"
WALKED_FROM_COIN = "WALKED_FROM_COIN"
DROP_BOMB_NEXT_TO_CRATE = "DROP_BOMB_NEXT_TO_CRATE"
WALKED_FROM_DANGER = "WALKED_FROM_DANGER"
STAYS_IN_DANGER_ZONE = "STAYS_IN_DANGER_ZONE"
HAS_NO_ESCAPE = "HAS_NO_ESCAPE"


# This is only an example!
# Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 1  # keep only ... last transitions
# ALPHA = 0.1
# BATCH_SIZE = 2000
GAMMA = 0.5  # 0.5 seems good
SAMPLE_PROP = 0.1  # proportion of data each tree is fitted on in percent

# for convenience
A_NUM = 6
A_IDX = np.arange(0, A_NUM, 1, dtype="int")
ALPHA = 0.05


def setup_training(self):
    # self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.reward_data = 0  # for plotting

    with open("data.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "score", "survival time", "total rewards"])


def tabular_Q_update(self, idx, old_feat, new_feat, reward):
    used_Q_values = Q_func(self, old_feat)
    used_Q_value = used_Q_values[idx]
    next_Q_value = Q_func(self, new_feat)
    Y = ALPHA * (reward + GAMMA * np.max(next_Q_value))

    self.Q_dicts[idx][tuple(old_feat)] = used_Q_value + Y
    # print(self.Q_dicts[idx][tuple(old_feat)])

    # if np.max(self.Q_dicts[idx][tuple(old_feat)]) != 0:
    #     self.Q_dicts[idx][tuple(old_feat)] = self.Q_dicts[idx][tuple(old_feat)] / np.max(self.Q_dicts[idx][tuple(old_feat)])


def game_events_occurred(
    self,
    old_game_state: dict,
    self_action: str,
    new_game_state: dict,
    events: List[str],
):

    # custom events use event_functions here
    if old_game_state is not None:
        old_feat = state_to_features(old_game_state)
        new_feat = state_to_features(new_game_state)
        if walked_towards_closest_coin(old_feat, self_action):
            events.append(WALKED_TO_COIN)
        if walked_towards_closest_coin(old_feat, self_action):
            events.append(WALKED_FROM_COIN)
        if walked_from_danger(old_feat, self_action) == 1:
            events.append(WALKED_FROM_DANGER)
        elif walked_from_danger(old_feat, self_action) == -1:
            events.append(STAYS_IN_DANGER_ZONE)
        if drop_bomb_next_to_crate(old_feat, events):
            events.append(DROP_BOMB_NEXT_TO_CRATE)
        if has_no_escape(new_feat, self_action):
            events.append(HAS_NO_ESCAPE)

        rewards = reward_from_events(self, events)
        self.reward_data += rewards

        idx = A_TO_NUM[self_action]
        # print('feat: ', state_to_features(new_game_state))
        # self.logger.debug(f"feature_vec: {old_feat}")
        tabular_Q_update(self, idx, old_feat, new_feat, rewards)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    with open("data.csv", "a") as f:
        writer = csv.writer(f)
        writer.writerow(
            (
                last_game_state["round"],
                last_game_state["self"][1],
                last_game_state["step"],
                self.reward_data,
            )
        )

    reward = total_rewards(self, events, last_game_state, None)
    self.reward_data += reward

    # reset reward counter for plotting
    self.reward_data = 0

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.Q_dicts, file)
    # with open("current_model/feature_history.pt", "wb") as file:
    #     pickle.dump(self.feat_history, file)
    # with open("current_model/next_feature_history.pt", "wb") as file:
    #     pickle.dump(self.next_feat_history, file)
    # with open("current_model/reward_history.pt", "wb") as file:
    #     pickle.dump(self.reward_history, file)


def total_rewards(self, events, old_game_state, new_game_state):
    rewards = [reward_from_events(self, events)]
    return sum(rewards)
