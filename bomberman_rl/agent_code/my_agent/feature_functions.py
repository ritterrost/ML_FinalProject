"""
  Input: game_state: {
    'round': int,
    'step': int.
    'field': np.array(width, height),
    'bombs': [(int, int), int],
    'explosion_map': np.array(wifht, height)
    'coins': [(x, y)],
    'self': (str, int, bool, (int, int)),
    'others': [(str, int, bool, (int, int))],
    'user_input': str|None
  }
  Output: features: np.ndarray()
"""

import numpy as np
from agent_code.my_agent import callbacks

# Feature Parameter
COIN_K = 1
BOMBS_FEAT_SIZE = 12

ENV = np.array([[-1,0],[1,0],[0,-1],[0,1]]) # up, down, left, right

def state_to_features_coin_collector(game_state):
    # Gather information about the game state
    arena = game_state["field"]
    explosion_map = game_state["explosion_map"]
    coins = np.asarray(game_state["coins"])
    _, score, bombs_left, (x, y) = game_state["self"]
    bombs = game_state["bombs"]
    # not used yet
    others = [xy for (n, s, b, xy) in game_state["others"]]

    #auxilliary features
    player_pos = np.array((x,y))
    surrounding = tuple((player_pos + ENV).T)
    obstacles = np.logical_or(arena[surrounding], explosion_map[surrounding])
    wall_above, wall_below, wall_left, wall_right = obstacles

    # turn bombs into constant sized feature
    # only consider bombs in 'range'
    # bomb_xys = [[bomb_x-x, bomb_y-y, bomb_t] for ((bomb_x, bomb_y), bomb_t) in bombs]
    # if len(bomb_xys) > 1:
    #     bomb_mask = np.logical_and(np.abs(bomb_xys[:,0])<RANGE, np.abs(bomb_xys[:,1])<RANGE)
    #     bomb_xys = np.array(bomb_xys[:,bomb_mask]).flatten()
    #     num_missing_bombs = BOMBS_FEAT_SIZE - bomb_xys.size
    #     bomb_feat = np.concatenate((bomb_xys, np.zeros(num_missing_bombs)))

    # else:
    #     bomb_feat = np.zeros(BOMBS_FEAT_SIZE)

    # only give direction of K coins
    is_no_coins = False

    if len(coins) == 0:
        is_no_coins = True
        coins_feat = np.zeros((COIN_K, 2)).flatten()
    else:
        coin_xy_rel = coins - np.asarray((x, y))
        sorted_index = np.argsort(np.linalg.norm(coin_xy_rel, axis=1))[:COIN_K]
        coins_feat = coin_xy_rel[sorted_index]

    coins_feat = np.sign(coins_feat)
    # exclude directions with walls
    if not is_no_coins:
        if wall_above:
            mask = coins_feat[:, 1] == 1
            coins_feat[mask, 1] = 0
        if wall_below:
            mask = coins_feat[:, 1] == -1
            coins_feat[mask, 1] = 0
        if wall_right:
            mask = coins_feat[:, 0] == 1
            coins_feat[mask, 0] = 0
        if wall_left:
            mask = coins_feat[:, 0] == -1
            coins_feat[mask, 0] = 0

    if not is_no_coins and np.all(coins_feat == 0):
        # choose random direction without wall
        available_directions = []
        if not wall_above:
            available_directions.append([0, 1])
        if not wall_below:
            available_directions.append([0, -1])
        if not wall_right:
            available_directions.append([1, 0])
        if not wall_left:
            available_directions.append([-1, 0])
        if len(available_directions) > 0:
            chosen_direction = np.random.randint(len(available_directions))
            coins_feat[0, 0], coins_feat[0, 1] = (
                np.array(available_directions)[chosen_direction, 0],
                np.array(available_directions)[chosen_direction, 1],
            )

    coins_feat = coins_feat.flatten()

    # turn bombs left into int32 numpy array with
    bombs_left = (np.asarray(bombs_left)).astype("int32")

    # construct feature vector
    feature_vec = np.concatenate((obstacles, coins_feat))
    return feature_vec
