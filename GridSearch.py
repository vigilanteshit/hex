import multiprocessing
import itertools
import Agent
import math
import csv
import os

def gridsearch_job_listener(m_queue):
    
    path_csv = './training_runs/training_log.csv'

    logfile_exists = os.path.isfile(path_csv) # If file exists do not write header again

    
    with open(path_csv, 'a') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter='|', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        logging_parameter = ['training_run_id', 'number_of_episodes', 'learning_rate', 'similarity_panelty', 'similarity_penalty_decay_rate', 'move_similarity_penalty', 'gradient_clipping_parameter', 'number_of_training_games', 'number_of_training_games_won', 'number_of_eval_games', 'base_eval_number_of_games_won', 'base_average_moves_to_win', 'pred_eval_number_of_games_won', 'pred_average_moves_to_win']
        if not logfile_exists:
            csvwriter.writerow(logging_parameter)
            
        while True:
            print("\nTuning Job Listener Alive and Well\n")
            item = m_queue.get()
            if item == 'kill':
                print('Training Job Listener: Process killed')
                break
            csvwriter.writerow(item)


def gridsearch_job(board_size, training_run_id:str, number_of_jobs:int, arg_list:list, m_queue):
    
    with multiprocessing.Pool(number_of_jobs) as pool:
        print(f"\nNumber of parallel jobs: {number_of_jobs}\n")
        processes = [pool.apply_async(agent._gridsearch_job, (board_size, training_run_id, episodes, learning_r, similarity_p, move_similarity_p, similarity_penalty_decay, random_c, gradient_c, number_of_eval_g, m_queue)) for episodes, learning_r, similarity_p, move_similarity_p, similarity_penalty_decay, random_c, gradient_c, number_of_eval_g, agent in arg_list]
        [p.get() for p in processes]
        
        # kill listener
        m_queue.put('kill')
        
def run_gridsearch_job(board_size, training_run_id:str, number_of_jobs:int, number_of_episodes:list, learning_rate:list, similarity_panelty:list, move_similarity_penalty:list, similarity_penalty_decay_rate:list, random_choice:list, gradient_clipping_parameter:list, number_of_eval_games:list):
    
    argument_list = list(itertools.product(number_of_episodes, learning_rate, similarity_panelty, move_similarity_penalty, similarity_penalty_decay_rate, random_choice, gradient_clipping_parameter, number_of_eval_games))
    argument_list_len = len(argument_list)
    
    print(f'\n{argument_list_len} models to train\n')
    
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
    
    gridsearch_job(board_size, training_run_id, number_of_jobs, argument_list_with_agents, queue) 
    
    listener.join()