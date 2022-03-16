def setup(self):

    pass


def act(self, game_state: dict):
    self.logger.info("Pick action according to pressed key")
    self.logger.debug(f"{game_state['self'][3]}")
    self.logger.debug(
        f"field size: {game_state['field'].shape}, expl size: {game_state['explosion_map'].shape}"
    )
    return game_state["user_input"]
