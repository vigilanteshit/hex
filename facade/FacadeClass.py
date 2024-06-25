from group_i import Agent

BOARD_SIZE = 7

final_model_path = './group_i/7x7_Convnet_9ca6ded8-31b0-11ef-bb9f-3e22fb4c01bd_conv_net_.keras'

class Facade:

    def __init__(self):
        self.hex_agent = Agent.REINFORCE_Agent(board_size=BOARD_SIZE)
        self.hex_agent.load_model_eval(path_to_model=final_model_path)

    def agent(self):
        next_action = self.hex_agent.machine()
        return next_action
        

