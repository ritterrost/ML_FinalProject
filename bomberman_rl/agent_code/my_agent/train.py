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

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"
ALPHA = 0.1
BATCH_SIZE = 10


def TD_target_respones(self, gamma=0.1):
    transition = self.transitions.pop()

    # Q function von next wert
    # next_feat = state_to_features(transition.next_state)
    next_q_value = Q_func(self, transition.next_state)

    return transition.reward + gamma * np.max(next_q_value)


def gradient_update(self, idx):

    if len(self.feat_history[idx]) >= BATCH_SIZE:
        selection_size = len(self.target_history[idx])
        selection_mask = np.random.permutation(np.arange(selection_size))[0:BATCH_SIZE]

        Y_tt = np.array(self.target_history[idx])[selection_mask]
        feats = np.array(self.feat_history[idx])[selection_mask]
        RHS = np.sum(np.sum(feats * (Y_tt[:, None] - feats * self.betas[idx]), axis=1))
        self.betas[idx] = self.betas[idx] + (ALPHA / BATCH_SIZE) * RHS

    else:
        Y_tt = np.array(self.target_history[idx])
        feats = np.array(self.feat_history[idx])
        RHS = np.sum(np.sum(feats * (Y_tt[:, None] - feats * self.betas[idx]), axis=1))
        self.betas[idx] = self.betas[idx] + (ALPHA / BATCH_SIZE) * RHS


def setup_training(self):
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)


def game_events_occurred(
    self,
    old_game_state: dict,
    self_action: str,
    new_game_state: dict,
    events: List[str],
):

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

        Y_tt = TD_target_respones(self, 1)
        idx = A_TO_NUM[self_action]

        self.target_history[A_TO_NUM[self_action]].append(Y_tt)
        self.feat_history[idx].append(feat)
        self.logger.info(f"self.target_history: {self.target_history}")
        self.logger.info(f"self.feat_history: {self.feat_history}")

        gradient_update(self, idx)


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

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.betas, file)


def reward_from_events(self, events: List[str]) -> int:
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.INVALID_ACTION: -1
        # e.KILLED_OPPONENT: 2,
        # PLACEHOLDER_EVENT: -0.1,  # idea: the custom event is bad
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
