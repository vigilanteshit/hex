import hex_engine

class HexEnv():

    def __init__(self, size=7):
        self.game = hex_engine.hexPosition(size)
        self.action_space = self.game.get_action_space()
        self.terminated = False
        self.truncated = False
        self.info = None

    def step(self, action):
        '''
        Run one timestep of the environment’s dynamics using the agent actions. When the end of an episode is reached 
        (terminated or truncated), it is necessary to call reset() to reset this environment’s state for the next episode.
        
            Parameters:
                action:Tuple A tuple representing the next move avaialble in self.action_space
            
            Returns:
                self.game.board:list[list[int]] An array representing the hex board. '0' means empty. '1' means 'white'. '-1' means 'black'.
                reward:int An integer representing a reward: 0 for every move that does not end the game. 1/-1 depending on if black or white has won
                self.terminated:Bool True if the game is over
                self.truncated:Bool not implemented -> always False
                self.info:Bool not implemented -> always None
        '''
        self.game.moove(action)
        self.game.evaluate(verbose=False)

        reward = self.game.winner
        if self.game.winner != 0:
            self.terminated = True

        self.action_space = self.game.get_action_space()

        return self.game.board, reward, self.terminated, self.truncated, self.info

    def reset(self):
        '''
        This method resets the game 
        '''
        self.game.reset()
        self.action_space = self.game.get_action_space()
        self.terminated = False
        self.truncated = False
        self.info = None

    def flip_board(self):
        self.game.recode_black_as_white()
