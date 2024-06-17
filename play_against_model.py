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
    
    agent.policy_net = agent.load_model('/Users/wolfgangwilke/RL/hex/hex/training_runs/Baseline9_96def192-2cc0-11ef-9025-acde48001122/Baseline9_96def192-2cc0-11ef-9025-acde48001122_dense_nn_net_.keras')
    agent2.policy_net = agent.load_model('/Users/wolfgangwilke/RL/hex/hex/training_runs/Baseline7_f69296c4-2c9e-11ef-8965-acde48001122/Baseline7_f69296c4-2c9e-11ef-8965-acde48001122_conv_net_.keras')
    
    #game.human_vs_machine(machine=agent2.machine)
    game.machine_vs_machine(machine1=agent2.machine, machine2=agent.machine)
    
