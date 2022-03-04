from time import sleep

from scipy.fft import next_fast_len
from agent_code.my_agent.callbacks import A_TO_NUM, Q_func
from collections import namedtuple, deque
import pickle
from typing import List
from agent_code.tpl_agent.callbacks import ACTIONS
import events as e
from agent_code.my_agent.callbacks import state_to_features, BATCH_SIZE
import numpy as np


# This is only an example!
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 1  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"
ALPHA = 0.1
GAMMA = 1


def immediate_q_update(self, idx):
    transition = self.transitions[-1]
    next_q_value = Q_func(self, transition.next_state)
    # Y_tt = transition.reward + np.max(next_q_value)
    # print("Y_tt: ", Y_tt)
    print("self.Q_pred: ", self.Q_pred[idx])
    self.betas[idx] += ALPHA * (
        transition.reward + GAMMA * np.max(next_q_value) - self.Q_pred[idx]
    )


def TD_target_func(self):
    transition = self.transitions[-1]
    next_q_value = Q_func(self, transition.next_state)
    return transition.reward + GAMMA * np.max(next_q_value)


def step_gradient_update(self, idx):
    selection_mask = np.random.permutation(np.arange(BATCH_SIZE))[0:BATCH_SIZE]
    # print("selection mask: ", selection_mask)
    targets = self.target_history.get_by_list(idx, selection_mask)
    # print("targets: ", targets)
    feats = self.feat_history.get_by_list(idx, selection_mask)
    # print("feats: ", feats)
    sum = 0
    beta = self.betas[idx]

    for i in range(BATCH_SIZE):
        # print("beta.shape: ", beta.shape)
        # print("feats[i].shape: ", feats[i].shape)
        # print("targets[i]: ", targets[i])
        print(
            "targets[i] - np.dot(feats[i], beta)", targets[i] - np.dot(feats[i], beta)
        )
        np.dot(feats[i], beta)
        sum += np.dot(feats[i], (targets[i] - np.dot(feats[i], beta)))

    self.betas[idx] += (ALPHA / BATCH_SIZE) * sum
    self.betas[idx] += self.betas[idx]
    # print("self_betas: ", self.betas[idx])


def round_gradient_update(self):
    for i, act in enumerate(ACTIONS):
        selection_mask = np.random.permutation(np.arange(BATCH_SIZE))[0:BATCH_SIZE]
        targets = self.target_history.get_by_list(i, selection_mask)
        feats = self.feat_history.get_by_list(i, selection_mask)
        sum = 0
        beta = self.betas[i]

        for j in range(BATCH_SIZE):
            # print("beta.shape: ", beta.shape)
            # print("feats[i].shape: ", feats[i].shape)
            # print("targets[i]: ", targets[i])
            # print(
            #     "targets[i] - np.dot(feats[i], beta)",
            #     targets[i] - np.dot(feats[i], beta),
            # )
            np.dot(feats[j], beta)
            sum += np.dot(feats[j], (targets[j] - np.dot(feats[j], beta)))

        self.betas[i] += (ALPHA / BATCH_SIZE) * sum
        self.betas[i] += self.betas[i]


def setup_training(self):
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)


def game_events_occurred(
    self,
    old_game_state: dict,
    self_action: str,
    new_game_state: dict,
    events: List[str],
):
    logging = True

    # Idea: Add your own events to hand out rewards
    if ...:
        events.append(PLACEHOLDER_EVENT)

    # state_to_features is defined in callbacks.py
    if old_game_state is not None:
        self.transitions.append(
            Transition(
                state_to_features(old_game_state),
                self_action,
                state_to_features(new_game_state),
                reward_from_events(self, events),
            )
        )
        feat = state_to_features(old_game_state)
        Y_tt = TD_target_func(self)
        idx = A_TO_NUM[self.transitions[-1].action]
        self.target_history.append(idx, Y_tt)
        self.feat_history.append(idx, feat)

        # if self.feat_history.get_storage_size(idx) > BATCH_SIZE:
        #     print("update")
        #     gradient_update(self, idx)
        if logging:
            # self.logger.info(f"self.transitions: {self.transitions}")
            self.logger.info(
                f"self.target_history: {self.target_history.get_storage()}"
            )
            # self.logger.info(f"self.feat_history: {self.feat_history.get_storage()}")
            self.logger.info(f"self.betas: {self.betas}")
            # immediate_q_update(self, idx)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    self.logger.debug(
        f'Encountered event(s) {", ".join(map(repr, events))} in final step'
    )
    self.transitions.append(
        Transition(
            state_to_features(last_game_state),
            last_action,
            None,
            reward_from_events(self, events),
        )
    )

    round_gradient_update(self)

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.betas, file)


def reward_from_events(self, events: List[str]) -> int:
    game_rewards = {
        e.COIN_COLLECTED: 2,
        e.INVALID_ACTION: -1
        # e.KILLED_OPPONENT: 2,
        # PLACEHOLDER_EVENT: -0.1,  # idea: the custom event is bad
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    return reward_sum
