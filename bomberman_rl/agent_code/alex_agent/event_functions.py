"""
  Input: self, events, old_game_state, new_game_state
  Output: Amount of reward: int
"""

import numpy as np
import events as e


def walked_towards_closest_coin(self, events, old_game_state, new_game_state):
    old_pos, new_pos = old_game_state["self"][-1], new_game_state["self"][-1]
    old_arena = old_game_state["field"]

    # surrounding walls
    wall_above = old_arena[old_pos[0], old_pos[1] + 1] == -1
    wall_below = old_arena[old_pos[0], old_pos[1] - 1] == -1
    wall_right = old_arena[old_pos[0] + 1, old_pos[1]] == -1
    wall_left = old_arena[old_pos[0] - 1, old_pos[1]] == -1

    # position of coins
    old_coins, new_coins = old_game_state["coins"], new_game_state["coins"]
    if len(old_coins) == 0:
        return 0
    elif len(new_coins) == 0:
        return 0
    else:
        old_coin_xy, new_coin_xy = np.asarray(old_coins), np.asarray(new_coins)
        # old closest coin
        old_closest_index = np.argmin(np.linalg.norm(old_coin_xy - old_pos, axis=1))
        old_coin_pos = old_coin_xy[old_closest_index]
        # direction of coin relative to old position
        coin_direction = old_coin_pos - old_pos
        # exclude case where player has to move laterally
        lateral_movement_necessary = (
            (np.all(coin_direction == [0, 1]) and wall_above)
            or (np.all(coin_direction == [0, -1]) and wall_below)
            or (np.all(coin_direction == [1, 0]) and wall_right)
            or (np.all(coin_direction == [-1, 0]) and wall_left)
        )
        if lateral_movement_necessary:
            return 0.5
        # check if there still is a coin at position of old closest coin
        if e.COIN_COLLECTED in events:
            return 1
        if old_coin_pos in new_coin_xy:
            # give reward if walked in direction, punish if walked away
            diff = np.linalg.norm(old_coin_pos - new_pos) - np.linalg.norm(
                old_coin_pos - old_pos
            )
            return np.sign(-diff)  # +/-1 after 1 step
        else:
            return 0
