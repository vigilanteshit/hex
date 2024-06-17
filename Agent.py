import PolicyNet
import hex_engine
import tensorflow as tf
import heapq
import numpy as np
import keras
import math
import datetime
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.random import choice
import csv
import uuid
import time
import tqdm
import ast  
import re



SIMILARITY_PENALTY = -0.3  # 0 means no penalty
MOVE_SIMILARITY_PENALTY = -0.07 # means no penalty
SIMILARITY_PENALTY_DECAY_RATE = 0.01 # 0 means no decay
RANDOM_CHOICE = 0.3 # 0 means no randomness in oponents playing
GRADIENT_CLIPPING_PARAMETER = 0.5
NUMBER_OF_EVALUATION_GAMES = 10
BREVITY_ENCOURAGEMENT = 0.009



class REINFORCE_Agent():
    
    def __init__(self, board_size):
        self.board_size = board_size
        self.num_fields = board_size**2
        self.policy_net = None
        self.number_of_games_won = 0
        self.number_of_games_trained = 0
        self.ratio_won_to_played = []
        self.number_of_episodes = 0
        self.number_of_training_layers = 0
        self.number_of_iterations = 0
        self.eval_random_won = 0
        self.eval_baseline_won = 0
        self.layer_config = ""


    
    def get_next_action(self, game:hex_engine.hexPosition, action_space:list, policy_prediction:np.array) -> tuple:
        
        best_actions_sorted = self._find_indices(policy_prediction)
        action_space = list(map(game.coordinate_to_scalar, action_space))
        
        for best_action in best_actions_sorted:
            if best_action in action_space:
                return game.scalar_to_coordinates(best_action), best_action
        
        return None
    
    def get_next_action_play(self, action_space:list, policy_prediction:np.array) -> tuple:
    
        best_actions_sorted = self._find_indices(policy_prediction)
        action_space = list(map(self.coordinate_to_scalar, action_space))
        
        for best_action in best_actions_sorted:
            if best_action in action_space:
                return self.scalar_to_coordinates(best_action), best_action
        
        return None
    
    def learn_random(self, num_episodes:int, learning_rate:float, game:hex_engine.hexPosition, policy_net:keras.Model, reward_type:str) -> tuple:
        '''
        This method learns by playing against an oponent that exclusively executes random moves.
        
            Parameters:
                num_episodes:int The number of episodes (aka games) to be played
                learning_rate:float The learning rate
                game:hex_engine.hexPosition An instance of the hex engine
                
            Returns:
                A tuple containing the number of games played, the number of games won and an array containing the ratio of each episode

        '''
        self.policy_net = policy_net
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipvalue=GRADIENT_CLIPPING_PARAMETER)
        
        for episode in range(num_episodes):
            
            episode_states = []
            episode_actions = []
            episode_rewards = []
            action_log_probabilites = []
            returns = []
            policy_loss = []
            number_of_moves = 0
            
            game.reset()
            
            
            with tf.GradientTape() as tape:
                while True:
                    
                    # Convert the board (aka the current game configuration or state) to a tf tensor of shape (board_size, board_size, 1)
                    board_tensor = tf.convert_to_tensor(game.board, dtype=tf.float32)
                    game_state_tensor = tf.reshape(board_tensor, (1, self.board_size , self.board_size , 1))
                    
                    # forward pass
                    model_output = self.policy_net(game_state_tensor) #output shape (1, board_size)
                    log_model_output = tf.math.log(model_output)
                  
                    # select next action 
                    next_action, best_action_index = self.get_next_action(game, game.get_action_space(), model_output.numpy()[0])

                    # make a move
                    game.moove(next_action)
                    
                    episode_states.append(game_state_tensor)
                    episode_actions.append(next_action)
                
                    action_log_probabilites.append(log_model_output[0,best_action_index])
             
                    if game.winner != 0:
                        episode_rewards.append(self.get_reward(episode_actions, game, reward_function=reward_type))
                        number_of_moves += 1
                        break
                    else:
                        number_of_moves += 1
                        game._random_moove()
                        game.evaluate()

                        if game.winner != 0:
                            episode_rewards.append(self.get_reward(episode_actions, game, reward_function=reward_type))
                            break
                        else:
                            episode_rewards.append(self.get_reward(episode_actions, game, reward_function=reward_type))
                
                if game.winner == 1:
                    self.number_of_games_won += 1
                          
                print(f"Process ID {os.getpid()}: Learning Strategy: Random Play - White") 
                # calculate returns    
                G = 0 
                for reward in reversed(episode_rewards):
                    G = reward + G
                    returns.insert(0, G)
                
                # generate loss
                for log_prob, G in zip(action_log_probabilites, returns):
                    policy_loss.append(-log_prob * G * learning_rate)

            self.number_of_games_trained += 1
            
            # calculate gradients and update model      
            grads = tape.gradient(policy_loss, self.policy_net.trainable_weights)    
            optimizer.apply(grads, self.policy_net.trainable_weights)
            
            self.ratio_won_to_played.append(self.number_of_games_won/self.number_of_games_trained)

        #self.save_model()
        return self.number_of_games_trained, self.number_of_games_won, self.ratio_won_to_played  
    
    def learn_selfplay(self, num_episodes:int, learning_rate:float, game:hex_engine.hexPosition, policy_net:keras.Model, random_choice, reward_type:str): 
        '''
        This methods lets the agent learn by playing against itself.
        
            Parameters:
                num_episodes:int The number of episodes (aka games) to be played
                learning_rate:float The learning rate
                game:hex_engine.hexPosition An instance of the hex engine
            
            Returns:
                A tuple containing the number of games played, the number of games won and an array containing the ratio of each episode
        '''
        actions_of_all_episodes = []
        
        self.policy_net = policy_net
        #self.policy_net = PolicyNet.create_hex_policy_net(self.board_size)
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipvalue=GRADIENT_CLIPPING_PARAMETER)
        
        for episode in range(num_episodes):

            
            flipped = False
            episode_states = []
            episode_actions = []
            episode_rewards = []
            action_log_probabilites = []
            returns = []
            policy_loss = []
            number_of_moves = 0
            
            game.reset()
     
            flipped = False
            
            
            
          
            with tf.GradientTape() as tape:
                while True:
                    
                    # Convert the board (aka the current game configuration or state) to a tf tensor of shape (board_size, board_size, 1)
                    board_tensor = tf.convert_to_tensor(game.board, dtype=tf.float32)
                    game_state_tensor = tf.reshape(board_tensor, (1, self.board_size , self.board_size , 1))
                    
                    # forward pass
                    model_output = self.policy_net(game_state_tensor) #output shape (1, board_size)
                    log_model_output = tf.math.log(model_output)
                
                    # select next action 
                    next_action, best_action_index = self.get_next_action(game, game.get_action_space(), model_output.numpy()[0])

                    # make a move
                    game.moove(next_action)

                    
                    episode_states.append(game_state_tensor)
                    episode_actions.append(next_action)
                
                    action_log_probabilites.append(log_model_output[0,best_action_index])
                    
                    game.evaluate()
            
                    if game.winner != 0:
                        episode_rewards.append(self.get_reward(episode_actions, game, reward_function=reward_type, flipped=flipped))
                        number_of_moves += 1
                        break
                    else: 
                        number_of_moves += 1
                        
                        flipped_board = game.recode_black_as_white()
                        flipped_board_tensor = tf.convert_to_tensor(game.recode_black_as_white(), dtype=tf.float32)
                        flipped_game_state_tensor = tf.reshape(flipped_board_tensor, (1, self.board_size , self.board_size , 1))
                        
                        # forward pass
                        model_output = self.policy_net(flipped_game_state_tensor) #output shape (1, board_size)
                        log_model_output = tf.math.log(model_output)
                    
                        # select next action 
                        next_action, best_action_index = self.get_next_action(game, game.get_action_space(), model_output.numpy()[0])

                        # make a move
                        if np.random.choice([True, False], 2, p=[random_choice, 1 - random_choice])[0]:
                            game._random_moove()
                        else:
                            game.moove(next_action)
                        
                        
                        game.evaluate()

                        if game.winner != 0:
                            episode_rewards.append(self.get_reward(episode_actions, game, reward_function=reward_type, flipped=flipped))
                            break
                        else:
                            episode_rewards.append(self.get_reward(episode_actions, game, reward_function=reward_type, flipped=flipped))
                
                if game.winner == 1:
                    self.number_of_games_won += 1
                    
                # Add similarity penalty
                actions_of_all_episodes.append(episode_actions)  
                episode_rewards[-1] =  episode_rewards[-1] + self.similarity_penalty(actions_of_all_episodes, SIMILARITY_PENALTY, True)
                
                # calculate returns    
                print(f"Process ID {os.getpid()}: Learning Strategy: Self Play - White")      
                G = 0 
                for reward in reversed(episode_rewards):
                    G = reward + G
                    returns.insert(0, G)
        
                
                # generate loss
                for log_prob, G in zip(action_log_probabilites, returns):
                    policy_loss.append(-log_prob * G * learning_rate)
                    
                #game._evaluate_white(verbose=True)

            
            self.number_of_games_trained += 1
            
            # calculate gradients and update model      
            grads = tape.gradient(policy_loss, self.policy_net.trainable_weights)    
            optimizer.apply(grads, self.policy_net.trainable_weights)
       
            
            self.ratio_won_to_played.append(self.number_of_games_won/self.number_of_games_trained)
            
        #self.save_model()
        return self.number_of_games_trained, self.number_of_games_won, self.ratio_won_to_played  

    def learn_selfplay_black(self, num_episodes:int, learning_rate:float, game:hex_engine.hexPosition, policy_net:keras.Model, random_choice, reward_type:str): 
        '''
        This methods lets the agent learn by playing against itself.
        
            Parameters:
                num_episodes:int The number of episodes (aka games) to be played
                learning_rate:float The learning rate
                game:hex_engine.hexPosition An instance of the hex engine
            
            Returns:
                A tuple containing the number of games played, the number of games won and an array containing the ratio of each episode
        '''
        actions_of_all_episodes = []
        flipped = False
        
        self.policy_net = policy_net
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipvalue=GRADIENT_CLIPPING_PARAMETER)
        
        for episode in range(num_episodes):
            
            flipped = False
            episode_states = []
            episode_actions = []
            episode_rewards = []
            action_log_probabilites = []
            returns = []
            policy_loss = []
            number_of_moves = 0
            
            game.reset()
     
            
            
            
            
          
            with tf.GradientTape() as tape:
                while True:
                    
                    # Convert the board (aka the current game configuration or state) to a tf tensor of shape (board_size, board_size, 1)
                    board_tensor = tf.convert_to_tensor(game.board, dtype=tf.float32)
                    game_state_tensor = tf.reshape(board_tensor, (1, self.board_size , self.board_size , 1))
                    
                    # forward pass
                    model_output = self.policy_net(game_state_tensor) #output shape (1, board_size)
                    log_model_output = tf.math.log(model_output)
                
                    # select next action 
                    next_action, best_action_index = self.get_next_action(game, game.get_action_space(), model_output.numpy()[0])
                    
                    
                    # make a move
                    if np.random.choice([True, False], 2, p=[random_choice, 1 - random_choice])[0]:
                        game._random_moove()
                    else:
                        game.moove(next_action)
                        
                    game.evaluate()
            
                    if game.winner != 0:
                        episode_rewards.append(self.get_reward(episode_actions, game, reward_function=reward_type, flipped=flipped))
                        number_of_moves += 1
                        break
                    else: # Flipped board
                        number_of_moves += 1
                        
                        flipped_board = game.recode_black_as_white()
                        flipped_board_tensor = tf.convert_to_tensor(game.recode_black_as_white(), dtype=tf.float32)
                        flipped_game_state_tensor = tf.reshape(flipped_board_tensor, (1, self.board_size , self.board_size , 1))
                        
                        # forward pass
                        model_output = self.policy_net(flipped_game_state_tensor) #output shape (1, board_size)
                        log_model_output = tf.math.log(model_output)
                    
                        # select next action 
                        next_action, best_action_index = self.get_next_action(game, game.get_action_space(), model_output.numpy()[0])
                        
                            
                        # make a move
                        game.moove(next_action)
                        episode_states.append(flipped_board_tensor)
                        episode_actions.append(next_action)
                
                        action_log_probabilites.append(log_model_output[0,best_action_index])

                        
                        game.evaluate()

                        if game.winner != 0:
                            episode_rewards.append(self.get_reward(episode_actions, game, reward_function=reward_type, flipped=flipped))
                            break
                        else:
                            episode_rewards.append(self.get_reward(episode_actions, game, reward_function=reward_type, flipped=flipped))
                
                if game.winner == 1:
                    self.number_of_games_won += 1
                    
                # Add similarity penalty
                actions_of_all_episodes.append(episode_actions)  
                episode_rewards[-1] =  episode_rewards[-1] + self.similarity_penalty(actions_of_all_episodes, SIMILARITY_PENALTY, True)
                
                # calculate returns    
                print(f"Process ID {os.getpid()}: Learning Strategy: Self Play - Black")          
                
                G = 0 
                for reward in reversed(episode_rewards):
                    G = reward + G
                    returns.insert(0, G)
        
                
                # generate loss
                for log_prob, G in zip(action_log_probabilites, returns):
                    policy_loss.append(-log_prob * G * learning_rate)
                
                #game._evaluate_black(verbose=True)
                    
        

            
            self.number_of_games_trained += 1
            
            # calculate gradients and update model      
            grads = tape.gradient(policy_loss, self.policy_net.trainable_weights)    
            optimizer.apply(grads, self.policy_net.trainable_weights)
          
            
            self.ratio_won_to_played.append(self.number_of_games_won/self.number_of_games_trained)

        #self.save_model()
        return self.number_of_games_trained, self.number_of_games_won, self.ratio_won_to_played   
    
    def learn_from_other_model(self, num_episodes:int, learning_rate:float, game:hex_engine.hexPosition, policy_net:keras.Model, random_choice, reward_type:str):
        '''
        
        This method lets the agent learn by playing against another model, possibly an older version of itself.
        
            Parameters:
                num_episodes:int The number of episodes (aka games) to be played
                learning_rate:float The learning rate
                game:hex_engine.hexPosition An instance of the hex engine
                model:keras.Model some other model
            
            Returns:
                A tuple containing the number of games played, the number of games won and an array containing the ratio of each episode
        '''
        actions_of_all_episodes = []
        flipped = False
        
        adveserial_net = REINFORCE_Agent(self.board_size)
        model_path = self._get_prev_model()
        adveserial_net.policy_net = adveserial_net.load_model(model_path)
        predecessor_policy_net = adveserial_net.policy_net 
        self.policy_net = policy_net
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        for episode in range(num_episodes):
            
            episode_states = []
            episode_actions = []
            episode_rewards = []
            action_log_probabilites = []
            returns = []
            policy_loss = []
            number_of_moves = 0
            
            game.reset()
            
            
            with tf.GradientTape() as tape:
                while True:
                    
                    # Convert the board (aka the current game configuration or state) to a tf tensor of shape (board_size, board_size, 1)
                    board_tensor = tf.convert_to_tensor(game.board, dtype=tf.float32)
                    game_state_tensor = tf.reshape(board_tensor, (1, self.board_size , self.board_size , 1))
                    
                    # forward pass
                    model_output = self.policy_net(game_state_tensor) #output shape (1, board_size)
                    log_model_output = tf.math.log(model_output)
                
                    # select next action 
                    next_action, best_action_index = self.get_next_action(game, game.get_action_space(), model_output.numpy()[0])

                    # make a move
                    game.moove(next_action)

                    
                    episode_states.append(game_state_tensor)
                    episode_actions.append(next_action)
                
                    action_log_probabilites.append(log_model_output[0,best_action_index])
                    
                    game.evaluate()
             
                    if game.winner != 0:
                        episode_rewards.append(self.get_reward(episode_actions, game, reward_function=reward_type, flipped=flipped))
                        number_of_moves += 1
                        break
                    else:
                        number_of_moves += 1
                        
                        flipped_board = game.recode_black_as_white()
                        flipped_board_tensor = tf.convert_to_tensor(game.recode_black_as_white(), dtype=tf.float32)
                        flipped_game_state_tensor = tf.reshape(flipped_board_tensor, (1, self.board_size , self.board_size , 1))
                        
                        # forward pass
                        model_output = predecessor_policy_net(flipped_game_state_tensor) #output shape (1, board_size)
                        log_model_output = tf.math.log(model_output)
                    
                        # select next action 
                        next_action, best_action_index = self.get_next_action(game, game.get_action_space(), model_output.numpy()[0])

                        # make a move
                        if np.random.choice([True, False], 2, p=[random_choice, 1 - random_choice])[0]:
                            game._random_moove()
                        else:
                            game.moove(next_action)
                        
                        game.evaluate()

                        if game.winner != 0:
                            episode_rewards.append(self.get_reward(episode_actions, game, reward_function=reward_type, flipped=flipped))
                            break
                        else:
                            episode_rewards.append(self.get_reward(episode_actions, game, reward_function=reward_type, flipped=flipped))
                
                if game.winner == 1:
                    self.number_of_games_won += 1
                    
                # Add similarity penalty
                actions_of_all_episodes.append(episode_actions)  
                episode_rewards[-1] =  episode_rewards[-1] + self.similarity_penalty(actions_of_all_episodes, SIMILARITY_PENALTY, True)
                
                print(f"Process ID {os.getpid()}: Learning Strategy: Other Model - White") 
                #game.print()        
                # calculate returns    
                G = 0 
                for reward in reversed(episode_rewards):
                    G = reward + G
                    returns.insert(0, G)
                
                # generate loss
                for log_prob, G in zip(action_log_probabilites, returns):
                    policy_loss.append(-log_prob * G * learning_rate)
                    
        
            self.number_of_games_trained += 1
            
            # calculate gradients and update model      
            grads = tape.gradient(policy_loss, self.policy_net.trainable_weights)    
            optimizer.apply(grads, self.policy_net.trainable_weights)
            
            self.ratio_won_to_played.append(self.number_of_games_won/self.number_of_games_trained)


        #self.save_model()
        return self.number_of_games_trained, self.number_of_games_won, self.ratio_won_to_played   
    
    def learn_from_other_model_black(self, num_episodes:int, learning_rate:float, game:hex_engine.hexPosition, policy_net:keras.Model, random_choice, reward_type:str):
        '''
        
        This method lets the agent learn by playing against another model, possibly an older version of itself.
        
            Parameters:
                num_episodes:int The number of episodes (aka games) to be played
                learning_rate:float The learning rate
                game:hex_engine.hexPosition An instance of the hex engine
                model:keras.Model some other model
            
            Returns:
                A tuple containing the number of games played, the number of games won and an array containing the ratio of each episode
        '''
        actions_of_all_episodes = []
        flipped = False
        
        adveserial_net = REINFORCE_Agent(self.board_size)
        model_path = self._get_prev_model()
        adveserial_net.policy_net = adveserial_net.load_model(model_path)
        predecessor_policy_net = adveserial_net.policy_net 
        self.policy_net = policy_net
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        for episode in range(num_episodes):
            
            episode_states = []
            episode_actions = []
            episode_rewards = []
            action_log_probabilites = []
            returns = []
            policy_loss = []
            number_of_moves = 0
            
            game.reset()
            
            
            with tf.GradientTape() as tape:
                while True:
                    
                    # Convert the board (aka the current game configuration or state) to a tf tensor of shape (board_size, board_size, 1)
                    board_tensor = tf.convert_to_tensor(game.board, dtype=tf.float32)
                    game_state_tensor = tf.reshape(board_tensor, (1, self.board_size , self.board_size , 1))
                    
                    # forward pass
                    model_output = predecessor_policy_net(game_state_tensor) #output shape (1, board_size)
                    log_model_output = tf.math.log(model_output)
                
                    # select next action 
                    next_action, best_action_index = self.get_next_action(game, game.get_action_space(), model_output.numpy()[0])
                    
                    if np.random.choice([True, False], 2, p=[random_choice, 1 - random_choice])[0]:
                        game._random_moove()
                    else:
                        game.moove(next_action)
                        
                    # make a move
           
                    
                    game.evaluate()
             
                    if game.winner != 0:
                        episode_rewards.append(self.get_reward(episode_actions, game, reward_function=reward_type, flipped=flipped))
                        number_of_moves += 1
                        break
                    else:
                        number_of_moves += 1
                        
                        flipped_board = game.recode_black_as_white()
                        flipped_board_tensor = tf.convert_to_tensor(game.recode_black_as_white(), dtype=tf.float32)
                        flipped_game_state_tensor = tf.reshape(flipped_board_tensor, (1, self.board_size , self.board_size , 1))
                        
                        # forward pass
                        model_output = self.policy_net(flipped_game_state_tensor) #output shape (1, board_size)
                        log_model_output = tf.math.log(model_output)
                    
                        # select next action 
                        next_action, best_action_index = self.get_next_action(game, game.get_action_space(), model_output.numpy()[0])
                        
                        game.moove(next_action)
                        episode_states.append(flipped_board_tensor)
                        episode_actions.append(next_action)
                        
                        action_log_probabilites.append(log_model_output[0,best_action_index])

    
                        
                        game.evaluate()

                        if game.winner != 0:
                            episode_rewards.append(self.get_reward(episode_actions, game, reward_function=reward_type, flipped=flipped))
                            break
                        else:
                            episode_rewards.append(self.get_reward(episode_actions, game, reward_function=reward_type, flipped=flipped))
                
                if game.winner == 1:
                    self.number_of_games_won += 1
                    
                # Add similarity penalty
                actions_of_all_episodes.append(episode_actions)  
                episode_rewards[-1] =  episode_rewards[-1] + self.similarity_penalty(actions_of_all_episodes, SIMILARITY_PENALTY, True)
                
                print(f"Process ID {os.getpid()}: Learning Strategy: Other Model - Black") 
                #game.print()        
                # calculate returns    
                G = 0 
                for reward in reversed(episode_rewards):
                    G = reward + G
                    returns.insert(0, G)
                
                # generate loss
                for log_prob, G in zip(action_log_probabilites, returns):
                    policy_loss.append(-log_prob * G * learning_rate)
                    
        
            self.number_of_games_trained += 1
            
            # calculate gradients and update model      
            grads = tape.gradient(policy_loss, self.policy_net.trainable_weights)   
            optimizer.apply(grads, self.policy_net.trainable_weights)
            
            self.ratio_won_to_played.append(self.number_of_games_won/self.number_of_games_trained)


        #self.save_model()
        return self.number_of_games_trained, self.number_of_games_won, self.ratio_won_to_played   
                 
    def save_model(self):
        '''
        This method saves the policy net of the agent. 
        '''
        
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = 'model_dump'
        if not os.path.isdir(path):
            os.mkdir(path)
        path = os.path.join(path,f'{self.policy_net.name}{current_time}.keras')
        self.policy_net.save(path) 
            
    def load_model(self, path_to_model:str):     
        return keras.models.load_model(path_to_model)
    

    def play_human_vs_machine(self, game:hex_engine.hexPosition, model_path:str):
        '''
        IGNORE FOR NOW
        ''' 
        while game.winner == 0:

            self.policy_net = self.load_model(model_path)
            board_tensor = tf.convert_to_tensor(game.board, dtype=tf.float32)
            game_state_tensor = tf.reshape(board_tensor, (1, self.board_size , self.board_size , 1))
            
            # forward pass
            model_output = self.policy_net(game_state_tensor) #output shape (1, board_size)  
            # select next action 
            next_action, best_action_index = self.get_next_action(game, game.get_action_space(recode_black_as_white=True), model_output.numpy()[0])
            
            game.moove(next_action)
            
            flipped_board = game.recode_black_as_white()
            game.recode_coordinates
        
    def _find_indices(self, np_predictions:np.array) -> np.array:
        '''
        Helper function
        '''
        
        return np.flip(np.argsort(np_predictions))
    
    def get_reward(self, episode_actions, game:hex_engine.hexPosition, reward_function=None, flipped=False) -> int:
        '''
        This method modifies the reward structure in order to encourge or discourage certain behaviors
        
            Parameters:
                episode_actions: list[tuple] A list of actions played in previous games
                game:hex_engine.hexPosition An instance of the hex game
        '''
        
        if reward_function == None:
            return game.winner
        else:
            game.evaluate()
            if game.winner != 0:
                return game.winner
            else:
                if reward_function == 'short':
                    return self.reward_encourage_shorter_games(episode_actions, BREVITY_ENCOURAGEMENT, game)
                if reward_function == 'explore':
                    return self._reward_encourage_exlporation(episode_actions, MOVE_SIMILARITY_PENALTY, game)
                if reward_function == 'reversed-short':
                    return self.reward_encourage_shorter_games_reversed(episode_actions, BREVITY_ENCOURAGEMENT, game)
                if reward_function == 'reversed-explore':
                    return self._reward_encourage_exlporation_reveresed(episode_actions, MOVE_SIMILARITY_PENALTY, game)
                if reward_function == 'explore-growth':
                    return self._reward_encourage_exlporation_grwoth(episode_actions, MOVE_SIMILARITY_PENALTY, game)
                if reward_function == 'explore-growth-reversed':
                    return self._reward_encourage_exlporation_reveresed_growth(episode_actions, MOVE_SIMILARITY_PENALTY, game)
                
    def _reward_encourage_exlporation(self, episode_actions: list[tuple], penalty:float, game:hex_engine.hexPosition):
        '''
        This method is to be used in the reward function. This method encourges the white player to explore more vertically. It is penalized when the next move and, to a lesser extent, the one after that are horizontally.
        It seperatly penalizes if two times the same opening nove has been used.
            Parameters:
                episode_actions: list[tuple] A list of actions played in previous games
                penalty:float A small negative number
                game:hex_engine.hexPosition An instance of the hex game

        '''
        
        penal = 0
        
        # if len(episode_actions) == 2:
        #     if (episode_actions[-1][1] == episode_actions[-2][1]) and (episode_actions[-1][0] == episode_actions[-2][0]):
        #         penal = -0.001 
            
            
        if len(episode_actions) > 2:
            if (abs(episode_actions[len(episode_actions) - 1 - 1][1] - episode_actions[len(episode_actions) - 1 - 2][1]) == 2):
                penal = penal + penalty
        if len(episode_actions) > 3:
            if (abs(episode_actions[len(episode_actions) - 1 - 1][1] - episode_actions[len(episode_actions) - 1 - 3][1]) == 3):
                penal = penal + penalty 

        return penal + game.winner
    
    def _reward_encourage_exlporation_grwoth(self, episode_actions: list[tuple], penalty:float, game:hex_engine.hexPosition):
        '''
        This method is to be used in the reward function. This method encourges the white player to explore more vertically. It is penalized when the next move and, to a lesser extent, the one after that are horizontally.
        It seperatly penalizes if two times the same opening nove has been used.
            Parameters:
                episode_actions: list[tuple] A list of actions played in previous games
                penalty:float A small negative number
                game:hex_engine.hexPosition An instance of the hex game

        '''
        
        penal = 0
        
        # if len(episode_actions) == 2:
        #     if (episode_actions[-1][1] == episode_actions[-2][1]) and (episode_actions[-1][0] == episode_actions[-2][0]):
        #         penal = -0.001 
            
            
        if len(episode_actions) > 2:
            if (abs(episode_actions[len(episode_actions) - 1 - 1][1] - episode_actions[len(episode_actions) - 1 - 2][1]) == 2):
                penal = penal + penalty * self.similarity_penalty_exponential_growth(self.number_of_iterations , rate=0.001)[self.number_of_games_trained]
        if len(episode_actions) > 3:
            if (abs(episode_actions[len(episode_actions) - 1 - 1][1] - episode_actions[len(episode_actions) - 1 - 3][1]) == 3):
                penal = penal + penalty * self.similarity_penalty_exponential_growth(self.number_of_iterations , rate=0.001)[self.number_of_games_trained]

        return penal + game.winner
    
    def _reward_encourage_exlporation_reveresed(self, episode_actions: list[tuple], penalty:float, game:hex_engine.hexPosition) -> float:
        '''
        This method is to be used in the reward function. This method encourges the black player to explore more vertically. It is penalized when the next move and, to a lesser extent, the one after that are horizontally.
        It seperatly penalizes if two times the same opening nove has been used.
            Parameters:
                episode_actions: list[tuple] A list of actions played in previous games
                penalty:float A small negative number
                game:hex_engine.hexPosition An instance of the hex game

        '''
        penal = 0
        
        # if len(episode_actions) == 2:
        #     if (episode_actions[-1][1] == episode_actions[-2][1]) and (episode_actions[-1][0] == episode_actions[-2][0]):
        #         penal = -0.5 
    
        if len(episode_actions) > 2:
            if (abs(episode_actions[len(episode_actions) - 1 - 1][1] - episode_actions[len(episode_actions) - 1 - 2][1]) == 2):
                penal = penalty * self.similarity_penalty_exponential_decay(self.number_of_iterations , rate=0.001)[self.number_of_games_trained]
        if len(episode_actions) > 3:
            if (abs(episode_actions[len(episode_actions) - 1 - 1][1] - episode_actions[len(episode_actions) - 1 - 3][1]) == 3):
                penal = penal + penalty  * self.similarity_penalty_exponential_decay(self.number_of_iterations , rate=0.001)[self.number_of_games_trained] 
        return penal  + game.winner
    
    def _reward_encourage_exlporation_reveresed_growth(self, episode_actions: list[tuple], penalty:float, game:hex_engine.hexPosition) -> float:
        '''
        This method is to be used in the reward function. This method encourges the black player to explore more vertically. It is penalized when the next move and, to a lesser extent, the one after that are horizontally.
        It seperatly penalizes if two times the same opening nove has been used.
            Parameters:
                episode_actions: list[tuple] A list of actions played in previous games
                penalty:float A small negative number
                game:hex_engine.hexPosition An instance of the hex game

        '''
        penal = 0
        
        # if len(episode_actions) == 2:
        #     if (episode_actions[-1][1] == episode_actions[-2][1]) and (episode_actions[-1][0] == episode_actions[-2][0]):
        #         penal = -0.5 
    
        if len(episode_actions) > 2:
            if (abs(episode_actions[len(episode_actions) - 1 - 1][1] - episode_actions[len(episode_actions) - 1 - 2][1]) == 2):
                penal = penalty * self.similarity_penalty_exponential_growth(self.number_of_iterations , rate=0.001)[self.number_of_games_trained]
        if len(episode_actions) > 3:
            if (abs(episode_actions[len(episode_actions) - 1 - 1][1] - episode_actions[len(episode_actions) - 1 - 3][1]) == 3):
                penal = penal + penalty  * self.similarity_penalty_exponential_growth(self.number_of_iterations , rate=0.001)[self.number_of_games_trained] 
        return penal  + game.winner
    
    def reward_encourage_shorter_games(self, episode_actions: list[tuple], penalty:float, game:hex_engine.hexPosition):

        number_of_moves = len(episode_actions)
        penal = number_of_moves * penalty
        return penal + game.winner
    
    def reward_encourage_shorter_games_reversed(self, episode_actions: list[tuple], penalty:float, game:hex_engine.hexPosition):

        number_of_moves = len(episode_actions)
        penal = number_of_moves * penalty
        return - penal + game.winner
    
    def similarity_penalty(self, actions_of_all_episodes:list[list], penalty_factor:float, decay=False):
        '''
        This method penalizes if similar moves are played in two consecutive games. The similarity is mesarued by the number of moves that appear in each of the episodes.
        
            Parameters:
                actions_of_all_episodes:list[list] A list of previouse played episodes
                penalty_factor:float A number that describes how heavy one move plyed in two consecutive games is pernalized.
                decay:bool If True the penalty decays exponentially with every played episode
        '''
        penal = 0
        if len(actions_of_all_episodes) > 1:
            for action in actions_of_all_episodes[-1]:
                if action in actions_of_all_episodes[-2]:
                    penal = penal + penalty_factor
                    
        if decay:
            penal = penal * self.similarity_penalty_exponential_decay(len(actions_of_all_episodes), SIMILARITY_PENALTY_DECAY_RATE)
        return penal
                
    def similarity_penalty_exponential_decay(self, trained_episodes:int, rate=0.1) -> float:
        '''
        This function goes exponentialy to zero over the interval of the number of training episodes. The steepness of decent is controlled by rate.
        
            Parameters:
            trained_episodes:int The number of training episodes
            rate:float Controls the steepness of decent in the interval [1, trained_episodes]

        '''
        return np.exp(-rate * np.arange(trained_episodes))
    
    def similarity_penalty_exponential_growth(self, trained_episodes:int, rate=0.1) -> float:
        '''
        This function goes exponentialy to zero over the interval of the number of training episodes. The steepness of decent is controlled by rate.
        
            Parameters:
            trained_episodes:int The number of training episodes
            rate:float Controls the steepness of decent in the interval [1, trained_episodes]

        '''
        return -np.exp(-rate * np.arange(trained_episodes)) + 1
    
    def evaluate_agent(self, another_player_path:str, game:hex_engine.hexPosition, number_of_games:int, baseline_model=False) -> tuple:
        '''
        This method evaluates the agents either agains itself or another model.
        
            Parameters: 
                another_player_path:str The path to another model or None
                ame:hex_engine.hexPosition A hex game
                number_of_games:int The number of time the models shall play
                self_play:bool If true the model plays against itself. In this case another_player_path can be None
                
            Returns:
                number_of_games_won:int The number of times the model has won the game
                average_won_in_moves:float  The average number of moves the model has won in        
                
        '''
        number_of_games_won = 0
        won_in_number_of_moves = []
        
        if baseline_model:
            adveserial_net = REINFORCE_Agent(self.board_size)
            adveserial_net.policy_net = adveserial_net.load_model(another_player_path)   
            print(f'Evaluating Agent - Previous Model {adveserial_net.policy_net.name}')
        
        print(f'\n\n******************* Process {os.getpid()} Evaluating Agent - Random Player *******************\n\n')
        print(f'******************* Process {os.getpid()} Number of Games {number_of_games} *******************\n\n')
        
        for games in range(number_of_games):
            game.reset()
            
            number_of_moves = 0
      
            
            while True:
                board_tensor = tf.convert_to_tensor(game.board, dtype=tf.float32)
                game_state_tensor = tf.reshape(board_tensor, (1, self.board_size , self.board_size , 1))
                
                # forward pass
                model_output = self.policy_net(game_state_tensor) #output shape (1, board_size)

                # select next action 
                next_action, best_action_index = self.get_next_action(game, game.get_action_space(), model_output.numpy()[0])

                # make a move
                
                game.moove(next_action)
                game.evaluate()
                
                if game.winner != 0:
                    number_of_moves += 1
                    if game.winner == 1:
                        #self.eval_random_won = self.eval_random_won + 1
                        won_in_number_of_moves.append(number_of_moves)
                    break
                
                else: 
                    number_of_moves += 1
                    
                    flipped_board_tensor = tf.convert_to_tensor(game.recode_black_as_white(), dtype=tf.float32)
                    flipped_game_state_tensor = tf.reshape(flipped_board_tensor, (1, self.board_size , self.board_size , 1))
                    
                    if baseline_model:
                    # forward pass
                        model_output = adveserial_net.policy_net(flipped_game_state_tensor) #output shape (1, board_size)
                    
                        # select next action 
                        next_action, best_action_index = self.get_next_action(game, game.get_action_space(), model_output.numpy()[0])

                        # make a move
                        game.moove(next_action)
                        
                    else:
                        game._random_moove()
                        
                        
                    game.evaluate()

                    if game.winner != 0:  
                        break
           
        
            if game.winner == 1:
                if baseline_model:
                    self.eval_baseline_won = self.eval_baseline_won + 1
                else:
                    self.eval_random_won = self.eval_random_won + 1
       
                
         
    def machine(self, board:list[list[int]], action_space:list[tuple]) -> tuple:
        '''
        This function is used when played human2machine or machine2machine. 
        
            Parameters:
                board:list[list[int]] The state of the board
                action_space:list[tuple] Available action space
        '''
        
        # Convert the board (aka the current game configuration or state) to a tf tensor of shape (board_size, board_size, 1)
        board_tensor = tf.convert_to_tensor(board, dtype=tf.float32)
        game_state_tensor = tf.reshape(board_tensor, (1, self.board_size , self.board_size , 1))
        
        # forward pass
        model_output = self.policy_net(game_state_tensor) #output shape (1, board_size)

        # select next action 
        next_action, best_action_index = self.get_next_action_play(action_space, model_output.numpy()[0])
        
        return next_action
    
    # Copied from hex script: needed for playing machine2machine
    def coordinate_to_scalar(self, coordinates):
        """
        Helper function to convert coordinates to scalars.
        This may be used as alternative coding for the action space.
        """
        assert (0 <= coordinates[0] and self.board_size - 1 >= coordinates[
            0]), "There is something wrong with the first coordinate."
        assert (0 <= coordinates[1] and self.board_size - 1 >= coordinates[
            1]), "There is something wrong with the second coordinate."
        return coordinates[0] * self.board_size + coordinates[1]
    # Copied from hex script: needed for playing machine2machine
    def scalar_to_coordinates(self, scalar):
        """
        Helper function to transform a scalar "back" to coordinates.
        Reverses the output of 'coordinate_to_scalar'.
        """
        coord1 = int(scalar / self.board_size)
        coord2 = scalar - coord1 * self.board_size
        assert (0 <= coord1 and self.board_size - 1 >= coord1), "The scalar input is invalid."
        assert (0 <= coord2 and self.board_size - 1 >= coord2), "The scalar input is invalid."
        return (coord1, coord2)
        
    def _gridsearch_job(self, board_size:int, training_run_id:str, number_of_episodes:int, learning_rate:float, brevity_encouragement=0.019, similarity_panelty=-0.003, move_similarity_penalty=-0.007, similarity_penalty_decay_rate=0.001, random_choice=0.3, gradient_clipping_parameter=0.5, number_of_eval_games=500, training_loop=None, policy_net=None, queue=None) -> None:
        '''
        This method is used to train a model. Several reward tuning parameters are available. How the model is trained also decided by which training layers (--> see below in the code) are present or removed.
        Once the model is trained, it is evaluated against a baseline model and in selfplay. The model is stored in the model dump but also in it training directory. The model dump can be used to get
        randomly older version of the model to play against a current model in training. 
        
            Paramters: 
                training_run_id:str A unique identifer for the training run
                number_of_episodes:int The number of training episodes per layer
                learning_rate:float The learning rate of the alogrithm
                game:hex_engine.hexPosition An instance of the hex game
                policy_net:keras.Model A policy net to be trained
        '''
  
        self.policy_net = policy_net
        self.number_of_training_episodes = number_of_episodes
        
        game = hex_engine.hexPosition(board_size)
        
        pid = os.getpid()
        time_based_uuid = str(uuid.uuid1())
        training_run_id = training_run_id + '_' + time_based_uuid
        
        SIMILARITY_PENALTY = similarity_panelty  # 0 means no penalty
        MOVE_SIMILARITY_PENALTY = move_similarity_penalty # means no penalty
        SIMILARITY_PENALTY_DECAY_RATE = similarity_penalty_decay_rate # 0 means no decay
        RANDOM_CHOICE = random_choice # 0 means no randomness in oponents playing
        GRADIENT_CLIPPING_PARAMETER = gradient_clipping_parameter
        NUMBER_OF_EVALUATION_GAMES = number_of_eval_games
        BREVITY_ENCOURAGEMENT = brevity_encouragement
                
        # If you want to change the baselinemodel, go ahead
        #BASELINE_MODEL_PATH = './model_dump/complex_hex_policy_conv_net_2024-06-01_21-27-52_good_config_sbrsbp_010624_09cb2fe8-204d-11ef-94f0-acde48001122.keras'
        BASELINE_MODEL_PATH = None
        
        
        ### file stuff --> DO NOT TOUCH ###
        # Model dump dir
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        dump_path = './model_dump'
        #dump_model_path = os.path.join(dump_path,f'{self.policy_net.name}{current_time}_{training_run_id}.keras')
        dump_model_path = os.path.join(dump_path,f'{training_run_id}_{self.policy_net.name}_FINAL.keras')
        
        # Create training_runs dir
        path = './training_runs'
        # if not os.path.isdir(path):
        #     os.mkdir(path)
            
        # Create training run dir
        path_training_runs = os.path.join(path,f'{training_run_id}')
        if not os.path.isdir(path_training_runs):
            os.mkdir(path_training_runs)
            
        # Save Model path dir
        #path_model = os.path.join(path_training_runs,f'{self.policy_net.name}{current_time}_{pid}.keras')
        #path_model_dir = os.path.join(path_training_runs, training_run_id)
        # if not os.path.isdir(path_model_dir):
        #     os.mkdir(path_model_dir)
        path_model = os.path.join(path_training_runs,f'{training_run_id}_{self.policy_net.name}.keras')

        
        # Save plots dir
        # path_plots_dir = os.path.join(path,f'{training_run_id}_plots')
        # if not os.path.isdir(path_plots_dir):
        #     os.mkdir(path_plots_dir)
        path_plots = os.path.join(path_training_runs,f'{training_run_id}_{self.policy_net.name}{current_time}_{pid}.png')
            
        random_old_model = self._get_random_old_model()
        #random_old_model_path = os.path.join(dump_path, random_old_model)
              
        predecessor_model_path = self._get_prev_model()
 
        ############################################# Execution of Training Loop - Start #################################################
        
        '''
        These trainig layers get executed sequentially and all train the same model
        
        '''
        
        self.extract_reward_structure(training_loop)
        
        self.number_of_episodes = self.sum_of_natural_numbers(number_of_episodes)
        
        percentage_10 = int(self.number_of_iterations /100 * 10)
        
        start = time.time()

        self.analyze_and_execute_training_loop(training_loop, 
                                               number_of_episodes, 
                                               game, 
                                               policy_net, 
                                               number_of_episodes, 
                                               percentage_10,
                                               learning_rate, 
                                               RANDOM_CHOICE,
                                               path_model)
        
        
        
                # played, won, ratio  = self.learn_selfplay(episode, learning_rate, game, policy_net, RANDOM_CHOICE, reward_type='explore')    # layer: s
                # played, won, ratio  = self.learn_selfplay_black(episode, learning_rate, game, policy_net, RANDOM_CHOICE) # layer: b
                # played, won, ratio  = self.learn_from_other_model(episode, learning_rate, game, random_old_model_path, self.policy_net, RANDOM_CHOICE) # layer: r
                # played, won, ratio  = self.learn_selfplay(episode, learning_rate, game, policy_net, RANDOM_CHOICE, reward_type='explore') # layer: s
                # played, won, ratio  = self.learn_selfplay_black(episode, learning_rate, game, policy_net, RANDOM_CHOICE) # layer: b
                # played, won, ratio  = self.learn_from_other_model(episode, learning_rate, game, predecessor_model_path, self.policy_net, RANDOM_CHOICE) # layer: p
                # played, won, ratio  = self.learn_from_other_model(episode, learning_rate, game, BASELINE_MODEL_PATH, self.policy_net, RANDOM_CHOICE)
                # played, won, ratio  = self.learn_from_other_model(episode, learning_rate, game, 'model_dump/simple_dense_nn_net_2024-06-01_21-49-43_dense_nn_psbrsbp_010624_1759b14a-2050-11ef-bdc1-acde48001122.keras', self.policy_net, RANDOM_CHOICE)
                # played, won, ratio  = self.learn_selfplay_black(episode, learning_rate, game, policy_net, RANDOM_CHOICE) # layer: b
                # played, won, ratio  = self.learn_from_other_model(episode, learning_rate, game, 'model_dump/simple_dense_nn_net_2024-06-01_23-10-51_loop20_dense_nn_psbrsbp_010624_6cfbff9e-205b-11ef-a149-acde48001122.keras', self.policy_net, RANDOM_CHOICE)
                # played, won, ratio  = self.learn_selfplay_black(episode, learning_rate, game, policy_net, RANDOM_CHOICE) # layer: b
                # played, won, ratio  = self.learn_from_other_model(episode, learning_rate, game, 'model_dump/complex_hex_policy_conv_net_2024-06-01_23-32-35_conc_22loop20_dense_nn_psbrsbp_010624_763bbe0c-205e-11ef-bb91-acde48001122.keras', self.policy_net, RANDOM_CHOICE)
                # played, won, ratio  = self.learn_from_other_model(episode, learning_rate, game, 'model_dump/simple_dense_nn_net_2024-06-02_00-09-19_dens_20conc_20loop20_dense_nn_psbrsbp_010624_97b4ef18-2063-11ef-9419-acde48001122.keras', self.policy_net, RANDOM_CHOICE)
       
        end = time.time()
        training_time = end - start
        ############################################# Execution of Training Loop - End #################################################

        # Save Trained Model --> DO NOT TOUCH
        self.policy_net.save(dump_model_path)
        self.policy_net.save(path_model)
        
        # Evaluate Model --> DO NOT TOUCH
        self.evaluate_agent(BASELINE_MODEL_PATH, game, NUMBER_OF_EVALUATION_GAMES)
        self.evaluate_agent(self._get_prev_model(), game, NUMBER_OF_EVALUATION_GAMES, self_play=True)
        
        # Write Parameters and Results in training_log --> DO NOT TOUCH
        row = [f'{training_run_id}', 
               f'{self.policy_net.name}',
               f'{self.number_of_training_layers}',  
               f'{self.layer_config}', 
               f'{self.extract_reward_structure(training_loop)}',
               f'{self.extract_training_loop_name(training_loop)}', 
               f'{training_time}', 
               f'{self.number_of_iterations}', 
               f'{learning_rate}', 
               f'{SIMILARITY_PENALTY}', 
               f'{SIMILARITY_PENALTY_DECAY_RATE}', 
               f'{MOVE_SIMILARITY_PENALTY}', 
               f'{BREVITY_ENCOURAGEMENT}', 
               f'{GRADIENT_CLIPPING_PARAMETER}',  
               f'{self.number_of_games_won}', 
               f'{NUMBER_OF_EVALUATION_GAMES}', 
               f'{self.eval_random_won}', 
               f'{self.eval_random_won/NUMBER_OF_EVALUATION_GAMES}', 
               f'{self.eval_baseline_won}', 
               f'{self.eval_baseline_won/NUMBER_OF_EVALUATION_GAMES}']
        
        # Put item in the queue
        queue.put(row)
        
        ratio = self.ratio_won_to_played
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(ratio)), ratio, marker='o', markersize=2)
        plt.xlabel('Games Played')
        plt.ylabel('Ratio: Won/Played (Training)')
        plt.title(f'Ratio of Won Games to Played Games\nlr:{learning_rate}/sp:{similarity_panelty}/msp:{move_similarity_penalty}/spd:{similarity_penalty_decay_rate}/rc:{random_choice}/gc:{gradient_clipping_parameter}')
        plt.grid(True)
        plt.savefig(path_plots)  # Save the plot as an image file
        plt.close()  
  

    def _get_prev_model(self) -> str:
        '''
        This method returns the path of the latest model in the model dump.
        '''
        model_dump_path = './model_dump'
        onlyfiles = [os.path.join(model_dump_path, f) for f in os.listdir(model_dump_path) if os.path.isfile(os.path.join(model_dump_path, f))]
        if onlyfiles:
            onlyfiles.sort(key=os.path.getctime) # sort descending
            return onlyfiles[-1]
        else:
            return None
    
    def _get_random_old_model(self) -> str:
        '''
        This method returns the path of a random model in model dump.
        '''
        import random
        model_dump_path = './model_dump'
        onlyfiles = [f for f in os.listdir(model_dump_path) if os.path.isfile(os.path.join(model_dump_path, f))]
        if not onlyfiles:
            return None
        return random.choice(onlyfiles)
    
     
    def analyze_and_execute_training_loop(self, loop_code:str, loop_iterations:int, game:hex_engine.hexPosition, policy_net:PolicyNet, number_of_episodes, percentage_10, learning_rate, RANDOM_CHOICE, path_model) -> None:
        '''
        This method determines the number of training layers in the code loop and executes the code loop. The number of training layers is needed
        to determine the total number of training episodes.
        
            Parameters: 
                loop_code:str A string containing a loop with training layers
                loop_iterations:int The number of training episodes per layer
                

        '''
        class MethodCallCounter(ast.NodeVisitor):
            def __init__(self):
                self.method_calls = {}
                self.methods_present = []

            def visit_Call(self, node):
                if isinstance(node.func, ast.Attribute):
                    method_name = node.func.attr
                    self.methods_present.append(method_name)
                    if method_name not in self.method_calls:
                        self.method_calls[method_name] = 0
                    self.method_calls[method_name] += 1
                self.generic_visit(node)
                
        tree = ast.parse(loop_code)

        # count number of training layers in training loop
        counter = MethodCallCounter()
        counter.visit(tree)
        total_calls = {method: count * loop_iterations for method, count in counter.method_calls.items()}
     
        self.layer_config = ','.join(counter.methods_present)
        
   
            
        self.number_of_training_layers = len(counter.methods_present)
        self.number_of_iterations = self.number_of_episodes * self.number_of_training_layers

        print(f'\n\n******************* Process {os.getpid()} Number of Training Layers: {self.number_of_training_layers} *******************\n\n')
        print(f'\n\n******************* Process {os.getpid()} Total Number of Training Episodes: {self.number_of_iterations} *******************\n\n')
        print(f'\n\n******************* Process {os.getpid()} Layer Config: {self.layer_config} *******************\n\n')
        
        # Execute the loop code
        exec(loop_code, {'self': self, 
                         'game': game, 
                         'policy_net': policy_net, 
                         'number_of_episodes':number_of_episodes, 
                         'percentage_10':percentage_10,
                         'learning_rate':learning_rate,
                         'RANDOM_CHOICE': RANDOM_CHOICE,
                         'path_model':path_model})
        
    def sum_of_natural_numbers(self, n:int) -> int:
        '''
        This method is simply an implementation of the guassian sum formula, to determine the number of training episodes executed by the training loop
        
            Parameters:
                n:int a natural number
            Returns:
                A natural number n + (n-1) + (n-2) + ... + (n - (n-1))
        '''
        if n <= 0:
            return 0
        return n * (n + 1) // 2
    
    def extract_training_loop_name(self,input_string):
        # Find the position of the first occurrence of the '#'
        hash_position = input_string.find('#')
        
        # If '#' is found, extract the substring after it
        if hash_position != -1:
            return input_string[hash_position + 1:].strip()
        else:
            return None  # Return None if '#' is not found
        
    def extract_reward_structure(self,input_string):
        # Define the regex pattern to match the specified words
        pattern = r"reward_type='(short|reversed-short|explore-growth|explore-growth-reversed|explore)'"
                 
        # Find all matches in the input string
        matches = re.findall(pattern, input_string)
        
        
        # Concatenate the matches into a single string separated by commas
        result = ", ".join(matches)
    
        
        return result



    
 
        
        
        
    
        
            
        
   
   
  
    
    
    
    