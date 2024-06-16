import multiprocessing
import itertools
import Agent
import math
import csv
import os
import PolicyNet
import tensorflow as tf

def gridsearch_job_listener(m_queue):
    
    path_csv = './training_runs/training_log.csv'

    logfile_exists = os.path.isfile(path_csv) # If file exists do not write header again

    
    with open(path_csv, 'a') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter='|', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        logging_parameter = ['training_run_id',
                             'policy_net', 
                             'number_of_layers',
                             'training_layer_config',
                             'reward_structure',
                             'loop_config_name', 
                             'training_time',
                             'number_of_episodes', 
                             'learning_rate', 
                             'similarity_panelty', 
                             'similarity_penalty_decay_rate', 
                             'move_similarity_penalty', 
                             'brevity_encouragment',
                             'gradient_clipping_parameter',  
                             'number_of_training_games_won', 
                             'number_of_eval_games', 
                             'eval_random_player_won_total', 
                             'eval_random_player_won_average', 
                             'eval_predecessor_agent_won', 
                             'average_eval_predecessor_agent_won']
        
        if not logfile_exists:
            csvwriter.writerow(logging_parameter)
            
        print("\n\n******************* Girdsearch Job Listener Alive *******************\n\n")  
          
        while True:
            #print("\nTuning Job Listener Alive and Well\n")
            item = m_queue.get()
            if item == 'kill':
                print('\n\n******************* Training Job Listener: Process killed *******************\n\n')
                break
            csvwriter.writerow(item)


def gridsearch_job(board_size, training_run_id:str, number_of_jobs:int, arg_list:list, m_queue):
    
    create_file_structure()
    
    with multiprocessing.Pool(number_of_jobs) as pool:
        
        print(f"\n\n******************* Process Pool Size: {number_of_jobs} *******************\n\n")
        
        processes = [pool.apply_async(agent._gridsearch_job, (board_size, 
                                                              training_run_id, 
                                                              episodes, 
                                                              learning_r, 
                                                              brevity,
                                                              similarity_p, 
                                                              move_similarity_p, 
                                                              similarity_penalty_decay, 
                                                              random_c, 
                                                              gradient_c, 
                                                              number_of_eval_g,
                                                              training_loop,
                                                              policy_net,  
                                                              m_queue)) 
                                                                for episodes, 
                                                                learning_r,
                                                                brevity, 
                                                                similarity_p, 
                                                                move_similarity_p, 
                                                                similarity_penalty_decay, 
                                                                random_c, 
                                                                gradient_c, 
                                                                number_of_eval_g, 
                                                                training_loop,
                                                                policy_net,
                                                                agent 
                                                                in arg_list]
        [p.get() for p in processes]
        
        # kill listener
        m_queue.put('kill')
        
def run_gridsearch_job(board_size, 
                       training_run_id:str, 
                       number_of_jobs:int,
                       policy_net:list, 
                       training_loop:list,
                       number_of_episodes:list, 
                       learning_rate:list, 
                       brevity_encouragment:list,
                       similarity_panelty:list, 
                       move_similarity_penalty:list, 
                       similarity_penalty_decay_rate:list, 
                       random_choice:list, 
                       gradient_clipping_parameter:list, 
                       number_of_eval_games:list):
    
        # Check if GPU is available
    if tf.config.list_physical_devices('GPU'):
        print("\n\n******************* Tensorflow: Using GPU *******************\n\n")
        for gpu in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
    elif tf.config.list_physical_devices('CPU'):
        print("\n\n******************* Tensorflow: Using CPU *******************\n\n")
    else:
        print("\n\n******************* Tensorflow: No compatible device found. Using default device. *******************\n\n")
    
    argument_list = list(itertools.product(number_of_episodes, 
                                           learning_rate,
                                           brevity_encouragment, 
                                           similarity_panelty, 
                                           move_similarity_penalty, 
                                           similarity_penalty_decay_rate, 
                                           random_choice, 
                                           gradient_clipping_parameter, 
                                           number_of_eval_games,
                                           training_loop,
                                           policy_net))
    
    argument_list_len = len(argument_list)
    
    print(f'\n\n******************* Model(s) to train: {argument_list_len}  *******************\n\n')
    
    agent_list = []
    for element in range(argument_list_len):
        agent_list.append(Agent.REINFORCE_Agent(board_size))
       
    argument_list_with_agents = [] 
    for arugment_set, agent in zip(argument_list, agent_list):
        argument_list_with_agents.append(arugment_set + (agent,))
 
    manager = multiprocessing.Manager()
    queue = manager.Queue()
    listener = multiprocessing.Process(target=gridsearch_job_listener, args=(queue,))
    listener.start()
    
    gridsearch_job(board_size, 
                   training_run_id, 
                   number_of_jobs, 
                   argument_list_with_agents, 
                   queue) 
    
    listener.join()
    
def create_file_structure():
    
    path = './training_runs'
    if not os.path.isdir(path):
        os.mkdir(path)
        
    dump_path = './model_dump'
    if not os.path.isdir(dump_path):
        os.mkdir(dump_path)
    
        
