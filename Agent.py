import PolicyNet
import hex_engine
import tensorflow as tf
import heapq
import numpy as np
import keras
import math
import datetime
import os


class REINFORCE_Agent():
    
    def __init__(self, board_size):
        self.board_size = board_size
        self.num_fields = board_size**2
        self.policy_net = None
        self.number_of_games_won = 0
        self.number_of_games_trained = 0
        self.ratio_won_to_played = []

    
    def get_next_action(self, game:hex_engine.hexPosition, action_space:list, policy_prediction:np.array) -> tuple:
        
        best_actions_sorted = self._find_indices(policy_prediction)
        action_space = list(map(game.coordinate_to_scalar, action_space))
        
        for best_action in best_actions_sorted:
            if best_action in action_space:
                return game.scalar_to_coordinates(best_action), best_action
        
        return None
  
        
    def learn_random(self, num_episodes:int, learning_rate:float, game:hex_engine.hexPosition, policy_net:keras.Model) -> tuple:
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
             
                    if game.winner != 0:
                        episode_rewards.append(self.get_reward(game))
                        number_of_moves += 1
                        break
                    else:
                        number_of_moves += 1
                        game._random_moove()
                        game.evaluate()

                        if game.winner != 0:
                            episode_rewards.append(self.get_reward(game))
                            break
                        else:
                            episode_rewards.append(self.get_reward(game))
                
                if game.winner == 1:
                    self.number_of_games_won += 1
                          
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
           
            if episode % 50 == 0:

                print(f'-------###  Games PLayed: {self.number_of_games_trained} ###------')
                print(f'  ratio won/played: {self.number_of_games_won/self.number_of_games_trained}')
                print("\n\n")

        self.save_model()
        return self.number_of_games_trained, self.number_of_games_won, self.ratio_won_to_played   

                
    def learn_selfplay(self, num_episodes:int, learning_rate:float, game:hex_engine.hexPosition, policy_net:keras.Model): 
        '''
        This methods lets the agent learn by playing against itself.
        
            Parameters:
                num_episodes:int The number of episodes (aka games) to be played
                learning_rate:float The learning rate
                game:hex_engine.hexPosition An instance of the hex engine
            
            Returns:
                A tuple containing the number of games played, the number of games won and an array containing the ratio of each episode
        '''
        self.policy_net = policy_net
        #self.policy_net = PolicyNet.create_hex_policy_net(self.board_size)
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
             
                    if game.winner != 0:
                        episode_rewards.append(self.get_reward(game))
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
                        game.moove(next_action)
                        
                        episode_states.append(game_state_tensor)
                        episode_actions.append(next_action)
                    
                        action_log_probabilites.append(log_model_output[0,best_action_index])
                        
                        game.evaluate()

                        if game.winner != 0:
                            episode_rewards.append(self.get_reward(game))
                            break
                        else:
                            episode_rewards.append(self.get_reward(game))
                
                if game.winner == 1:
                    self.number_of_games_won += 1
                          
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
           
            if episode % 50 == 0:

                print(f'---------- Games PLayed: {self.number_of_games_trained} ---------')
                print(f'  ratio won/played: {self.number_of_games_won/self.number_of_games_trained}')
                print("\n\n")


        self.save_model()
        return self.number_of_games_trained, self.number_of_games_won, self.ratio_won_to_played   
    
    def learn_from_other_model(self, num_episodes:int, learning_rate:float, game:hex_engine.hexPosition, model:keras.Model, policy_net:keras.Model):
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
        predecessor_policy_net = model
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
             
                    if game.winner != 0:
                        episode_rewards.append(self.get_reward(game))
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
                        game.moove(next_action)
                        
                        game.evaluate()

                        if game.winner != 0:
                            episode_rewards.append(self.get_reward(game))
                            break
                        else:
                            episode_rewards.append(self.get_reward(game))
                
                if game.winner == 1:
                    self.number_of_games_won += 1
                          
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
           
            if episode % 50 == 0:

                print(f'-------###  Games PLayed: {self.number_of_games_trained} ###------')
                print(f'  ratio won/played: {self.number_of_games_won/self.number_of_games_trained}')
                print("\n\n")

        self.save_model()
        return self.number_of_games_trained, self.number_of_games_won, self.ratio_won_to_played   
                 
    def save_model(self):
        ct = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = f'{ct}'
        if not os.path.isdir(path):
            os.mkdir(path)
        path = os.path.join(path,f'{self.policy_net.name}{ct}.keras')
        self.policy_net.save(path) 
        
        #self.policy_net.save(f'./{self.policy_net.name}{str(ct)[:-7]}.keras') 
    
    def load_model(self, path_to_model:str):
        return keras.models.load_model(path_to_model)
    
    def play_machine_vs_machine(self, board:list[list], action_space:list[tuple]):
        pass
    
    def _find_indices(self, np_predictions:np.array) -> np.array:
        
        return np.flip(np.argsort(np_predictions))
    
    def get_reward(self, game:hex_engine.hexPosition) -> int:
        game.evaluate()
        return game.winner
    
    def loss_function(self):
        pass
    
    