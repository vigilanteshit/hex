import hex_engine
import PolicyNet
import Agent
import tensorflow as tf
import matplotlib.pyplot as plt
from numpy.random import choice

NUMBER_OF_EPISODES = 1000
LEARNING_RATE = 1e-3
BOARD_SIZE = 3

if __name__ == '__main__':
   
    game = hex_engine.hexPosition(BOARD_SIZE) # Instantiate a hex game
     
    agent = Agent.REINFORCE_Agent(BOARD_SIZE) # Instantiate a REINFORCE_Agent
    agent2 = Agent.REINFORCE_Agent(BOARD_SIZE) # Instantiate a REINFORCE_Agent
    
    agent.policy_net = agent.load_model('/Users/wolfgangwilke/RL/hex/hex/training_runs/test4_eecbfc66-2f35-11ef-9dda-acde48001122/test4_eecbfc66-2f35-11ef-9dda-acde48001122_conv_net_.keras')
    agent2.policy_net = agent.load_model('/Users/wolfgangwilke/RL/hex/hex/training_runs/test5_9ae6fe5e-2f51-11ef-a032-acde48001122/test5_9ae6fe5e-2f51-11ef-a032-acde48001122_conv_net_.keras')
    
    game.human_vs_machine(machine=agent2.machine)
    #game.machine_vs_machine(machine1=agent.machine, machine2=agent2.machine)
    
