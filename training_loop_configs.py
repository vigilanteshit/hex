loop_code_baseline1 ='''for episode in range(number_of_episodes):
                
                if episode <= percentage_10:
                    
                    self.learn_random(episode, learning_rate, game, policy_net, reward_type='short')    
                    self.learn_selfplay_black(episode, learning_rate, game, policy_net, RANDOM_CHOICE, reward_type='reversed-short')            
                else:
                    self.learn_selfplay(episode, learning_rate, game, policy_net, RANDOM_CHOICE, reward_type='explore')
                    self.learn_selfplay_black(episode, learning_rate, game, policy_net, RANDOM_CHOICE, reward_type='reversed-explore')
                    #loop_code_baseline1
            '''
            
loop_code_baseline2 ='''for episode in range(number_of_episodes):
                
                self.learn_random(episode, learning_rate, game, policy_net, reward_type='short')    
                self.learn_selfplay_black(episode, learning_rate, game, policy_net, RANDOM_CHOICE, reward_type='reversed-short')            
                self.learn_selfplay(episode, learning_rate, game, policy_net, RANDOM_CHOICE, reward_type='explore')
                self.learn_selfplay_black(episode, learning_rate, game, policy_net, RANDOM_CHOICE, reward_type='reversed-explore')
                #loop_code_baseline2 
        '''
        
loop_code_baseline3 ='''for episode in range(number_of_episodes):
            
            if episode <= percentage_10:
        
                self.learn_random(episode, learning_rate, game, policy_net, reward_type='short')    
                self.learn_selfplay_black(episode, learning_rate, game, policy_net, RANDOM_CHOICE, reward_type='reversed-short')            
            else:
                self.learn_selfplay(episode, learning_rate, game, policy_net, RANDOM_CHOICE, reward_type='explore-growth')
                self.learn_selfplay_black(episode, learning_rate, game, policy_net, RANDOM_CHOICE, reward_type='explore-growth-reversed')
                #loop_code_baseline3
        '''
        
loop_code_baseline4 ='''for episode in range(number_of_episodes):
                
                self.learn_random(episode, learning_rate, game, policy_net, reward_type='short')    
                self.learn_selfplay_black(episode, learning_rate, game, policy_net, RANDOM_CHOICE, reward_type='reversed-short')            
                self.learn_selfplay(episode, learning_rate, game, policy_net, RANDOM_CHOICE, reward_type='explore-growth')
                self.learn_selfplay_black(episode, learning_rate, game, policy_net, RANDOM_CHOICE, reward_type='explore-growth-reversed')
                #loop_code_baseline4
        '''
