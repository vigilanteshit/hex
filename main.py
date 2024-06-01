import hex_engine
import PolicyNet
import Agent
import tensorflow as tf
import matplotlib.pyplot as plt
import GridSearch

NUMBER_OF_EPISODES = 2 # number of episodes per training layer
LEARNING_RATE = 1e-5
BOARD_SIZE = 7

if __name__ == '__main__':
    
                                ################# Example: Execute simple training run: pass policy net to the training method #################
   
    # get a hex game
    game = hex_engine.hexPosition(BOARD_SIZE) # Instantiate a hex game
    # Get policy net
    policy_net = PolicyNet.create_complex_hex_policy_net(BOARD_SIZE) # Instantiate some ploicy net
    print(policy_net.summary())
    # Get agent
    agent = Agent.REINFORCE_Agent(BOARD_SIZE) 
    # Start training run                       
    #agent.train('test_run5', NUMBER_OF_EPISODES, LEARNING_RATE, game, policy_net, similarity_panelty=-0.4, move_similarity_penalty=-0.07, similarity_penalty_decay_rate=0.1, random_choice=0.3, gradient_clipping_parameter=0.5, number_of_eval_games=10)
    



                                                        ################# Example: Execute Grid Search #################
    
    # Prepare parameter list
    number_of_episodes_list = [20]
    learning_rate_list = [1e-5]
    similarity_panelty_list = [-0.3, -0.4]
    move_similarity_penalty_list = [-0.07, 0.08]
    similarity_penalty_decay_rate_list = [0.01, 0.02]
    random_choice_list = [0.3, 0.4]
    gradient_clipping_parameter_list = [0.5, 0.4]
    number_of_eval_games_list = [25]
    
    # Training ID for the job
    training_run_base_id = 'concurrent_tuning_test4'

    # Execute GridSearch job: The job trains the models concurrently and saves them. 
    # Agents and policy nets are generate automatically along the way
    GridSearch.run_gridsearch_job(board_size=BOARD_SIZE,
                                training_run_id=training_run_base_id,
                                number_of_jobs=5, 
                                number_of_episodes=number_of_episodes_list, 
                                learning_rate=learning_rate_list,
                                similarity_panelty=similarity_panelty_list,
                                move_similarity_penalty=move_similarity_penalty_list,
                                similarity_penalty_decay_rate=similarity_penalty_decay_rate_list,
                                random_choice=random_choice_list,
                                gradient_clipping_parameter=gradient_clipping_parameter_list,
                                number_of_eval_games=number_of_eval_games_list)
    
        
      


    