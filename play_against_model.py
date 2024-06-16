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
    
    agent.policy_net = agent.load_model('model_dump/complex_hex_policy_conv_net_2024-06-02_00-35-30_conv20_dens_20conc_20loop20_dense_nn_psbrsbp_010624_40631e52-2067-11ef-bcf6-acde48001122.keras')
    agent2.policy_net = agent.load_model('/Users/wolfgangwilke/RL/hex/hex/model_dump/simple_dense_nn_net_2024-06-02_00-09-19_dens_20conc_20loop20_dense_nn_psbrsbp_010624_97b4ef18-2063-11ef-9419-acde48001122.keras')
    
    #game.human_vs_machine(machine=agent2.machine)
    game.machine_vs_machine(machine1=agent2.machine, machine2=agent.machine)
    
