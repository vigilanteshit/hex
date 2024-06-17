import hex_engine
import PolicyNet
import Agent
import tensorflow as tf
import matplotlib.pyplot as plt
import GridSearch
import training_loop_configs


NUMBER_OF_EPISODES = 2 # number of episodes per training layer
LEARNING_RATE = 1e-5
BOARD_SIZE = 7

if __name__ == '__main__':
   
                                                        ################# Example: Execute Grid Search #################

    
    # Prepare parameter list
    number_of_episodes_list = [40] # Number of episodes per training layer
    learning_rate_list = [1e-4]
    brevity_encouragment = [-0.1] # small positive number encourages shorter games, a negative number encourages longer games
    similarity_panelty_list = [-0.1] # small negative number encourages vertical exploration, positive number encourages horizontal moves
    move_similarity_penalty_list = [-0.1] # small negative number discourages playing similar moves in consecutive games
    similarity_penalty_decay_rate_list = [0.1, 0] # controls if and how the similarity penalty and move similarity penalty decreases/increases over the course of training
    random_choice_list = [0.3] # controls the likelihood of the opponent playing random moves
    gradient_clipping_parameter_list = [0.3]
    number_of_eval_games_list = [100] # the number of games for evaluation
    training_loop_list = [training_loop_configs.loop_code_advanced1]
    policy_net_list = [PolicyNet.create_complex_hex_policy_net(board_size=BOARD_SIZE)]
    


    # Give training ID for the job
    training_run_base_id = 'test'

    #Execute GridSearch job: The job trains the models concurrently and saves them. 
    #Agents and policy nets are generated automatically along the way
    GridSearch.run_gridsearch_job(board_size=BOARD_SIZE,
                                training_run_id=training_run_base_id,
                                number_of_jobs=5, 
                                policy_net=policy_net_list,
                                training_loop=training_loop_list,
                                number_of_episodes=number_of_episodes_list, 
                                learning_rate=learning_rate_list,
                                brevity_encouragment=brevity_encouragment,
                                similarity_panelty=similarity_panelty_list,
                                move_similarity_penalty=move_similarity_penalty_list,
                                similarity_penalty_decay_rate=similarity_penalty_decay_rate_list,
                                random_choice=random_choice_list,
                                gradient_clipping_parameter=gradient_clipping_parameter_list,
                                number_of_eval_games=number_of_eval_games_list)

        
      


    