import hex_engine
import PolicyNet
import Agent
import tensorflow as tf
import matplotlib.pyplot as plt
from numpy.random import choice

NUMBER_OF_EPISODES = 1000
LEARNING_RATE = 1e-3
BOARD_SIZE = 7

if __name__ == '__main__':
   
    game = hex_engine.hexPosition(BOARD_SIZE) # Instantiate a hex game
     
    agent = Agent.REINFORCE_Agent(BOARD_SIZE) # Instantiate a REINFORCE_Agent
    agent2 = Agent.REINFORCE_Agent(BOARD_SIZE) # Instantiate a REINFORCE_Agent
    
    agent.policy_net = agent.load_model('/Users/wolfgangwilke/RL/hex/hex/2024-05-31_10-48-13/complex_hex_policy_conv_net_2024-05-31_10-48-13.keras')
    agent2.policy_net = agent.load_model('/Users/wolfgangwilke/RL/hex/hex/2024-05-31_10-48-13/complex_hex_policy_conv_net_2024-05-31_10-48-13.keras')
    
    #game.human_vs_machine(machine=agent.machine)
    game.machine_vs_machine(machine1=agent.machine, machine2=agent.machine)
    
