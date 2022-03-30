import pickle
from typing import List
import numpy as np

from .event_functions import make_events
from .feature_functions import state_to_features_bfs_2 as state_to_features
from .feature_functions import update_batch
from .callbacks import A_TO_NUM, Q, ACTIONS
import events as e

import csv

# This is only an example!
# Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 1  # keep only ... last transitions
# ALPHA = 0.1
#BATCH_SIZE = 2000
GAMMA = 0.5 #0.5 seems good
SAMPLE_PROP = 0.1 #proportion of data each tree is fitted on in percent

# for convenience
A_NUM = 6
A_IDX = np.arange(0, A_NUM, 1, dtype="int")


def setup_training(self):
    # self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.reward_data = 0 # for plotting
    
    if not self.keep_training:
        with open('data.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['round', 'score', 'survival time', 'total rewards'])
        
        with open('bincount.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow([*(np.arange(0, 10) - 1), '>8']*4)
            writer.writerow([*["coin"]*11, *["crate"]*11, *["free"]*11, *["other"]*11, ])


def model_update(self):
    #select random batch of transitions and updates all actions at once
    for idx in A_IDX:
        if len(self.feat_history[idx])>0:
            X = np.array(self.feat_history[idx])
            next_q_value = np.array([Q(self, np.array(self.next_feat_history[idx]), a) for a in ACTIONS])
            # self.model[idx].set_params(max_samples=int(np.sqrt(len(X))))  #only choose 1/10 of dataset for fitting each tree
            y = np.array(self.reward_history[idx]) + GAMMA * np.max(next_q_value, axis=0)
            self.model[idx].fit(np.array(X),np.array(y))
        else:
            continue


def game_events_occurred(
    self,
    old_game_state: dict,
    self_action: str,
    new_game_state: dict,
    events: List[str],
):

    if old_game_state is not None:
        old_feat, new_feat, reward = make_events(self, old_game_state, new_game_state, 
                                                 self_action, events)
        self.reward_data += reward
        if e.INVALID_ACTION in events:
            self.logger.debug(f"Invalid action {self_action} was chosen from features {old_feat} and reward {reward}")    
        update_batch(self, old_feat, new_feat, reward, self_action)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    arrs = []
    for idx in A_IDX:
        if len(self.feat_history[idx]) == 0:
            continue
        else:
            arrs.append(np.array(self.feat_history[idx]))
    arr = np.vstack(arrs) + 1
    coin_dist = np.bincount(arr[:,2], minlength=11)
    crate_dist = np.bincount(arr[:,5], minlength=11)
    free_dist = np.bincount(arr[:,8], minlength=11)
    other_dist = np.bincount(arr[:,11], minlength=11)
    row = []
    for bincount in [coin_dist, crate_dist, free_dist, other_dist]:
        row.extend(bincount[:10])
        row.append(np.sum(bincount[10:]))
    with open('bincount.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(row)
    
    old_feat, new_feat, reward = make_events(self, last_game_state, None, 
                                             last_action, events)
    self.reward_data += reward
    with open('data.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow((last_game_state["round"], last_game_state["self"][1], last_game_state["step"], self.reward_data))

    # self.logger.debug(f"feature_vec: {old_feat}")
    update_batch(self, old_feat, new_feat, reward, last_action)

    # reset reward counter for plotting
    self.reward_data = 0

    model_update(self)

    # Store the model
    with open("current_model/my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)
    if (last_game_state['round'] % 10 == 9): # don't need to store the data every round
        with open("current_model/feature_history.pt", "wb") as file:
            pickle.dump(self.feat_history, file)
        with open("current_model/next_feature_history.pt", "wb") as file:
            pickle.dump(self.next_feat_history, file)
        with open("current_model/reward_history.pt", "wb") as file:
            pickle.dump(self.reward_history, file)
    
    self.logger.info(f"Death Note: {old_feat}, {last_action}, reward: {reward}")
    pass
