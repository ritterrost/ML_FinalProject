import collections
from operator import ne
from attr import field

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
SIGHT = 1
BOMBS_FEAT_SIZE = 12
BR = 3


def state_to_features_alex(game_state):
    # Gather information about the game state
    arena = game_state["field"]
    explosion_map = game_state["explosion_map"]
    coins = np.asarray(game_state["coins"])
    _, score, bombs_left, (x, y) = game_state["self"]
    bombs = game_state["bombs"]
    # not used yet
    others = [xy for (n, s, b, xy) in game_state["others"]]

    # surrounding walls
    wall_above = arena[x, y + 1] == -1
    wall_below = arena[x, y - 1] == -1
    wall_right = arena[x + 1, y] == -1
    wall_left = arena[x - 1, y] == -1

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
    # construct field feature
    xx, yy = np.clip(np.mgrid[x - SIGHT : x + SIGHT, y - SIGHT : y + SIGHT], 0, arena.shape[0] - 1)

    # relative field
    rel_field = np.array(arena[xx, yy]).flatten().astype("int32")

    # construct explosion map feature
    rel_exp_map = explosion_map[xx, yy].flatten().astype("int32")

    # turn bombs left into int32 numpy array with
    bombs_left = (np.asarray(bombs_left)).astype("int32")

    # construct feature vector
    feature_vec = np.concatenate((rel_field, coins_feat, rel_exp_map))
    return feature_vec


def state_to_features_bfs_2(game_state):
    arena = game_state["field"]
    others = np.asarray([xy for (n, s, b, xy) in game_state["others"]])
    coins = np.asarray(game_state["coins"])
    bombs = game_state["bombs"]
    bombs_xy = np.asarray([[bomb_x, bomb_y] for ((bomb_x, bomb_y), _) in bombs])
    explosion_map = game_state["explosion_map"]
    _, _, _, (x, y) = game_state["self"]
    pos_self = np.asarray((x, y))

    xx, yy = np.clip(np.mgrid[x - SIGHT : x + SIGHT, y - SIGHT : y + SIGHT], 0, arena.shape[0] - 1)
    rel_field = np.array(arena[xx, yy]).flatten().astype("int32")

    # overlay arena
    if others.shape[0] != 0:
        arena[others[:, 1], others[:, 0]] = 2

    if coins.shape[0] != 0:
        arena[coins[:, 1], coins[:, 0]] = 3

    # if bombs_xy.shape[0] != 0:
    #     arena[bombs_xy[:, 1], bombs_xy[:, 0]] = 4

    explosion_array = np.argwhere(explosion_map == 1)
    if explosion_array.shape[0] != 0:
        arena[explosion_array[:, 1], explosion_array[:, 0]] = 5

    is_in_danger_zone = False
    if bombs_xy.shape[0] != 0:
        for b in bombs_xy:
            b_xx, b_yy = np.clip(
                np.mgrid[b[0] - BR : b[0] + BR, b[1] - BR : b[1] + BR],
                0,
                arena.shape[0] - 1,
            )
            arena[b_xx, b_yy] = 4
            # danger_zone.append(np.asarray([b_xx, b_yy]))

    next_step = None
    if coins.shape[0] != 0:
        next_coord = bfs_cc(arena, pos_self, "coin")
        if next_coord is not None:
            next_step = pos_self - next_coord

    elif 1 in arena:
        next_coord = bfs_cc(arena, pos_self, "crate")
        if next_coord is not None:
            next_step = pos_self - next_coord

    if is_in_danger_zone:
        next_coord = bfs_cc(arena, pos_self, "free")

    is_in_danger_zone = np.asarray(is_in_danger_zone).astype(int)
    feature_vec = np.append(np.concatenate((rel_field, next_step)), is_in_danger_zone)
    return feature_vec


def state_to_features_bfs_cc(game_state):
    arena = game_state["field"]
    coins = np.asarray(game_state["coins"])
    _, _, _, (x, y) = game_state["self"]
    pos_self = np.asarray((x, y))
    arena_with_coins = arena

    if len(coins) != 0:
        arena_with_coins[coins[:, 1], coins[:, 0]] = 3
        next_corrd = bfs_cc(arena_with_coins, pos_self, "coin")
        next_step = pos_self - next_corrd
    else:
        next_step = np.zeros(2)

    # construct field feature
    xx, yy = np.clip(np.mgrid[x - SIGHT : x + SIGHT, y - SIGHT : y + SIGHT], 0, arena.shape[0] - 1)
    rel_field = np.array(arena[xx, yy]).flatten().astype("int32")

    feature_vec = np.concatenate((rel_field, next_step))
    # print("next step: ", next_step)
    return feature_vec


def bfs_cc(grid, start, target: str):
    target_dict = {
        "free": 0,
        "crate": 1,
        "coin": 3,
    }
    if target == "coin":
        block = 1
    else:
        block = -1
    goal = target_dict[target]
    queue = collections.deque([[start]])
    seen = set(start)
    while queue:
        path = queue.popleft()
        xy = path[-1]
        x = xy[0]
        y = xy[1]
        if grid[y][x] == goal:
            if len(path) > 1:
                return path[1]
            else:
                None
        for x2, y2 in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if (
                1 <= x2 < 16
                and 0 <= y2 < 16
                and grid[y2][x2] not in [-1, block, 2, 4, 5]
                and (x2, y2) not in seen
            ):
                queue.append(path + [(x2, y2)])
                seen.add((x2, y2))
