import hex_engine
import PolicyNet
import Agent
import tensorflow as tf
import matplotlib.pyplot as plt
import GridSearch
import training_loop_configs



if __name__ == '__main__':
   
                                                        ################# Example: Execute Grid Search #################

    
    # Prepare parameter list

                                                        ################# 1a) ResNet Base Training Run #################

    #BOARD_SIZE = 3

    # number_of_episodes_list = [20] # Number of episodes per training layer
    # learning_rate_list = [1e-3, 1e-4]
    # brevity_encouragment = [0.01] # small positive number encourages shorter games, a negative number encourages longer games
    # similarity_panelty_list = [-0.001, 0.0001] # small negative number encourages vertical exploration, positive number encourages horizontal moves
    # move_similarity_penalty_list = [-0.01, -0.001, 0.0001] # small negative number discourages playing similar moves in consecutive games
    # similarity_penalty_decay_rate_list = [0.1] # controls if and how the similarity penalty and move similarity penalty decreases/increases over the course of training
    # random_choice_list = [0.2] # controls the likelihood of the opponent playing random moves
    # gradient_clipping_parameter_list = [0.8]
    # number_of_eval_games_list = [100] # the number of games for evaluation
    # training_loop_list = [training_loop_configs.loop_code_baseline10,
    #                       training_loop_configs.loop_code_baseline11,
    #                       training_loop_configs.loop_code_baseline12]
    # policy_net_list = [PolicyNet.ResNet50(input_shape=(BOARD_SIZE, BOARD_SIZE, 1), classes=BOARD_SIZE*BOARD_SIZE)]

    
    # Give training ID for the job
    #training_run_base_id = 'ResNet50'

                                                        ################# 1b) ConvNet Base Training Run #################


    # number_of_episodes_list = [20] # Number of episodes per training layer
    # learning_rate_list = [1e-3, 1e-4]
    # brevity_encouragment = [0.01] # small positive number encourages shorter games, a negative number encourages longer games
    # similarity_panelty_list = [-0.001, 0.0001] # small negative number encourages vertical exploration, positive number encourages horizontal moves
    # move_similarity_penalty_list = [-0.01, -0.001, 0.0001] # small negative number discourages playing similar moves in consecutive games
    # similarity_penalty_decay_rate_list = [0.1] # controls if and how the similarity penalty and move similarity penalty decreases/increases over the course of training
    # random_choice_list = [0.2] # controls the likelihood of the opponent playing random moves
    # gradient_clipping_parameter_list = [0.8]
    # number_of_eval_games_list = [100] # the number of games for evaluation
    # training_loop_list = [training_loop_configs.loop_code_baseline10,
    #                       training_loop_configs.loop_code_baseline11,
    #                       training_loop_configs.loop_code_baseline12]
    # policy_net_list = [PolicyNet.create_complex_hex_policy_net(BOARD_SIZE)]
    
    # #Give training ID for the job
    # training_run_base_id = 'ConvNet'


                                                        ################# 1c) DenseNet Base Training Run #################


    # number_of_episodes_list = [20] # Number of episodes per training layer
    # learning_rate_list = [1e-3, 1e-4]
    # brevity_encouragment = [0.01] # small positive number encourages shorter games, a negative number encourages longer games
    # similarity_panelty_list = [-0.001, 0.0001] # small negative number encourages vertical exploration, positive number encourages horizontal moves
    # move_similarity_penalty_list = [-0.01, -0.001, 0.0001] # small negative number discourages playing similar moves in consecutive games
    # similarity_penalty_decay_rate_list = [0.1] # controls if and how the similarity penalty and move similarity penalty decreases/increases over the course of training
    # random_choice_list = [0.2] # controls the likelihood of the opponent playing random moves
    # gradient_clipping_parameter_list = [0.8]
    # number_of_eval_games_list = [100] # the number of games for evaluation
    # training_loop_list = [training_loop_configs.loop_code_baseline10,
    #                       training_loop_configs.loop_code_baseline11,
    #                       training_loop_configs.loop_code_baseline12]
    # policy_net_list = [PolicyNet.create_dense_nn_policy_net(BOARD_SIZE)]
    
    # #Give training ID for the job
    # training_run_base_id = 'DenseNet'

                                                        ################ 2a.1) Transfer Leaarning: Board Size 4x4 #################

    BOARD_SIZE = 4
    smaller_model_path = './training_runs/DenseNet_8f54a3e0-2fe5-11ef-a332-3e22fb4c01bd/DenseNet_8f54a3e0-2fe5-11ef-a332-3e22fb4c01bd_dense_net_.keras'

    #double episodes
    number_of_episodes_list = [40] # Number of episodes per training layer
    learning_rate_list = [1e-3]
    brevity_encouragment = [0.01] # small positive number encourages shorter games, a negative number encourages longer games
    similarity_panelty_list = [0.0001] # small negative number encourages vertical exploration, positive number encourages horizontal moves
    move_similarity_penalty_list = [-0.001] # small negative number discourages playing similar moves in consecutive games
    similarity_penalty_decay_rate_list = [0.1] # controls if and how the similarity penalty and move similarity penalty decreases/increases over the course of training
    random_choice_list = [0.2] # controls the likelihood of the opponent playing random moves
    gradient_clipping_parameter_list = [0.8]
    number_of_eval_games_list = [100] # the number of games for evaluation
    training_loop_list = [training_loop_configs.loop_code_baseline10]
    policy_net_list = [PolicyNet.load_model_for_generalizaton(larger_board_size=BOARD_SIZE, smaller_model_path=smaller_model_path)]
  
  
    #Give training ID for the job
    training_run_base_id = '4x4_DenseNet'

    #Execute GridSearch job: The job trains the models concurrently and saves them. 
    #Agents and policy nets are generated automatically along the way
    GridSearch.run_gridsearch_job(board_size=BOARD_SIZE,
                                training_run_id=training_run_base_id,
                                number_of_jobs=10, 
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
    

                                                     ################# 2a.2) Transfer Leaarning: ResNet - Board Size 5x5 #################


    # BOARD_SIZE = 5
    # smaller_model_path = './training_runs/4x4_DenseNet_d8090384-3255-11ef-be4a-3e22fb4c01bd/4x4_DenseNet_d8090384-3255-11ef-be4a-3e22fb4c01bd_conv_net_.keras'


    # number_of_episodes_list = [45] # Number of episodes per training layer
    # learning_rate_list = [1e-3]
    # brevity_encouragment = [0.01] # small positive number encourages shorter games, a negative number encourages longer games
    # similarity_panelty_list = [0.0001] # small negative number encourages vertical exploration, positive number encourages horizontal moves
    # move_similarity_penalty_list = [-0.001] # small negative number discourages playing similar moves in consecutive games
    # similarity_penalty_decay_rate_list = [0.1] # controls if and how the similarity penalty and move similarity penalty decreases/increases over the course of training
    # random_choice_list = [0.2] # controls the likelihood of the opponent playing random moves
    # gradient_clipping_parameter_list = [0.8]
    # number_of_eval_games_list = [100] # the number of games for evaluation
    # training_loop_list = [training_loop_configs.loop_code_baseline10]
    # policy_net_list = [PolicyNet.load_model_for_generalizaton(larger_board_size=BOARD_SIZE, smaller_model_path=smaller_model_path)]
  
    # # # #Give training ID for the job
    # training_run_base_id = '5x5_DenseNet'

    # #Execute GridSearch job: The job trains the models concurrently and saves them. 
    # #Agents and policy nets are generated automatically along the way
    # GridSearch.run_gridsearch_job(board_size=BOARD_SIZE,
    #                             training_run_id=training_run_base_id,
    #                             number_of_jobs=10, 
    #                             policy_net=policy_net_list,
    #                             training_loop=training_loop_list,
    #                             number_of_episodes=number_of_episodes_list, 
    #                             learning_rate=learning_rate_list,
    #                             brevity_encouragment=brevity_encouragment,
    #                             similarity_panelty=similarity_panelty_list,
    #                             move_similarity_penalty=move_similarity_penalty_list,
    #                             similarity_penalty_decay_rate=similarity_penalty_decay_rate_list,
    #                             random_choice=random_choice_list,
    #                             gradient_clipping_parameter=gradient_clipping_parameter_list,
    #                             number_of_eval_games=number_of_eval_games_list)

                                                    ################# 2a.3) Transfer Leaarning: ResNet - Board Size 6x6 #################

        
    # BOARD_SIZE = 6
    # smaller_model_path = './training_runs/5x5_Convnet_add149a2-3168-11ef-ba03-3e22fb4c01bd/5x5_Convnet_add149a2-3168-11ef-ba03-3e22fb4c01bd_conv_net_.keras'

    # number_of_episodes_list = [47] # Number of episodes per training layer
    # learning_rate_list = [1e-3]
    # brevity_encouragment = [0.01] # small positive number encourages shorter games, a negative number encourages longer games
    # similarity_panelty_list = [0.0001] # small negative number encourages vertical exploration, positive number encourages horizontal moves
    # move_similarity_penalty_list = [-0.001] # small negative number discourages playing similar moves in consecutive games
    # similarity_penalty_decay_rate_list = [0.1] # controls if and how the similarity penalty and move similarity penalty decreases/increases over the course of training
    # random_choice_list = [0.2] # controls the likelihood of the opponent playing random moves
    # gradient_clipping_parameter_list = [0.8]
    # number_of_eval_games_list = [100] # the number of games for evaluation
    # training_loop_list = [training_loop_configs.loop_code_baseline10]
    # policy_net_list = [PolicyNet.load_model_for_generalizaton(larger_board_size=BOARD_SIZE, smaller_model_path=smaller_model_path)]
  
  
    # # #Give training ID for the job
    # training_run_base_id = '6x6_Convnet'

    # #Execute GridSearch job: The job trains the models concurrently and saves them. 
    # #Agents and policy nets are generated automatically along the way
    # GridSearch.run_gridsearch_job(board_size=BOARD_SIZE,
    #                             training_run_id=training_run_base_id,
    #                             number_of_jobs=10, 
    #                             policy_net=policy_net_list,
    #                             training_loop=training_loop_list,
    #                             number_of_episodes=number_of_episodes_list, 
    #                             learning_rate=learning_rate_list,
    #                             brevity_encouragment=brevity_encouragment,
    #                             similarity_panelty=similarity_panelty_list,
    #                             move_similarity_penalty=move_similarity_penalty_list,
    #                             similarity_penalty_decay_rate=similarity_penalty_decay_rate_list,
    #                             random_choice=random_choice_list,
    #                             gradient_clipping_parameter=gradient_clipping_parameter_list,
    #                             number_of_eval_games=number_of_eval_games_list)


                                                        ################# 2a.3) Transfer Leaarning: ResNet - Board Size 7x7 #################

        
    # BOARD_SIZE = 7
    # smaller_model_path = './training_runs/6x6_Convnet_44f62a7e-3177-11ef-866a-3e22fb4c01bd/6x6_Convnet_44f62a7e-3177-11ef-866a-3e22fb4c01bd_conv_net_.keras'

    # number_of_episodes_list = [50] # Number of episodes per training layer
    # learning_rate_list = [1e-3]
    # brevity_encouragment = [0.01] # small positive number encourages shorter games, a negative number encourages longer games
    # similarity_panelty_list = [0.0001] # small negative number encourages vertical exploration, positive number encourages horizontal moves
    # move_similarity_penalty_list = [-0.001] # small negative number discourages playing similar moves in consecutive games
    # similarity_penalty_decay_rate_list = [0.1] # controls if and how the similarity penalty and move similarity penalty decreases/increases over the course of training
    # random_choice_list = [0.2] # controls the likelihood of the opponent playing random moves
    # gradient_clipping_parameter_list = [0.8]
    # number_of_eval_games_list = [100] # the number of games for evaluation
    # training_loop_list = [training_loop_configs.loop_code_baseline10]
    # policy_net_list = [PolicyNet.load_model_for_generalizaton(larger_board_size=BOARD_SIZE, smaller_model_path=smaller_model_path)]
  
    # #Give training ID for the job
    # training_run_base_id = '7x7_Convnet'

    # #Execute GridSearch job: The job trains the models concurrently and saves them. 
    # #Agents and policy nets are generated automatically along the way
    # GridSearch.run_gridsearch_job(board_size=BOARD_SIZE,
    #                             training_run_id=training_run_base_id,
    #                             number_of_jobs=10, 
    #                             policy_net=policy_net_list,
    #                             training_loop=training_loop_list,
    #                             number_of_episodes=number_of_episodes_list, 
    #                             learning_rate=learning_rate_list,
    #                             brevity_encouragment=brevity_encouragment,
    #                             similarity_panelty=similarity_panelty_list,
    #                             move_similarity_penalty=move_similarity_penalty_list,
    #                             similarity_penalty_decay_rate=similarity_penalty_decay_rate_list,
    #                             random_choice=random_choice_list,
    #                             gradient_clipping_parameter=gradient_clipping_parameter_list,
    #                             number_of_eval_games=number_of_eval_games_list)
