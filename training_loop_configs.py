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
        
loop_code_baseline5 ='''         
self.learn_random(number_of_episodes, learning_rate, game, policy_net, reward_type='short')   
self.learn_selfplay(number_of_episodes, learning_rate, game, policy_net, RANDOM_CHOICE, reward_type='explore') 
self.learn_selfplay_black(number_of_episodes, learning_rate, game, policy_net, RANDOM_CHOICE, reward_type='reversed-explore')  
self.learn_selfplay(number_of_episodes, learning_rate, game, policy_net, RANDOM_CHOICE, reward_type='short')     
self.learn_selfplay_black(number_of_episodes, learning_rate, game, policy_net, RANDOM_CHOICE, reward_type='reversed-short')     
#loop_code_baseline4
'''

loop_code_baseline6 ='''          
self.learn_selfplay(number_of_episodes, learning_rate, game, policy_net, RANDOM_CHOICE, reward_type='explore')  
self.learn_selfplay(number_of_episodes, learning_rate, game, policy_net, RANDOM_CHOICE, reward_type='short')   
#loop_code_baseline6
'''

loop_code_baseline7 ='''  
self.learn_selfplay(number_of_episodes, learning_rate, game, policy_net, RANDOM_CHOICE, None)         
self.learn_selfplay_black(number_of_episodes, learning_rate, game, policy_net, RANDOM_CHOICE, None)    
#loop_code_baseline7
'''

loop_code_baseline8 ='''  
self.learn_selfplay(number_of_episodes, learning_rate, game, policy_net, RANDOM_CHOICE, reward_type='explore')         
self.learn_selfplay_black(number_of_episodes, learning_rate, game, policy_net, RANDOM_CHOICE, reward_type='reversed-short')    
#loop_code_baseline8
'''
loop_code_baseline9 ='''  
self.learn_selfplay(number_of_episodes, learning_rate, game, policy_net, RANDOM_CHOICE, reward_type='explore')         
self.learn_selfplay_black(number_of_episodes, learning_rate, game, policy_net, RANDOM_CHOICE, reward_type='reversed-explore')    
#loop_code_baseline9
'''

loop_code_baseline10 ='''  
for episode in range(number_of_episodes):
    self.learn_selfplay(episode, learning_rate, game, policy_net, RANDOM_CHOICE, reward_type='explore')         
    self.learn_selfplay_black(episode, learning_rate, game, policy_net, RANDOM_CHOICE, reward_type='reversed-explore')  
    self.learn_selfplay(episode, learning_rate, game, policy_net, RANDOM_CHOICE, reward_type='short')         
    self.learn_selfplay_black(episode, learning_rate, game, policy_net, RANDOM_CHOICE, reward_type='reversed-short')   
#loop_code_baseline10
'''

loop_code_baseline11 ='''  
for episode in range(number_of_episodes):
    self.learn_selfplay(episode, learning_rate, game, policy_net, RANDOM_CHOICE, reward_type='explore')         
    self.learn_selfplay_black(episode, learning_rate, game, policy_net, RANDOM_CHOICE, reward_type='reversed-explore')  
    self.learn_selfplay(episode, learning_rate, game, policy_net, RANDOM_CHOICE, reward_type='explore')         
    self.learn_selfplay_black(episode, learning_rate, game, policy_net, RANDOM_CHOICE, reward_type='reversed-explore') 
#loop_code_baseline11
'''

loop_code_baseline12 ='''  
for episode in range(number_of_episodes):
    self.learn_selfplay(episode, learning_rate, game, policy_net, RANDOM_CHOICE, reward_type='explore')         
    self.learn_selfplay_black(episode, learning_rate, game, policy_net, RANDOM_CHOICE, reward_type='reversed-explore')  
    self.learn_selfplay(episode, learning_rate, game, policy_net, RANDOM_CHOICE, reward_type='explore')         
    self.learn_selfplay_black(episode, learning_rate, game, policy_net, RANDOM_CHOICE, reward_type='reversed-explore') 
#loop_code_baseline12
'''
        
loop_code_advanced1 ='''for episode in range(number_of_episodes):
                
                self.learn_from_other_model(episode, learning_rate, game, policy_net, RANDOM_CHOICE, reward_type='short')   
                self.learn_from_other_model_black(episode, learning_rate, game, policy_net, RANDOM_CHOICE, reward_type='reversed-short') 
                self.learn_selfplay(episode, learning_rate, game, policy_net, RANDOM_CHOICE, reward_type='short')
                self.learn_selfplay_black(episode, learning_rate, game, policy_net, RANDOM_CHOICE, reward_type='reversed-short')            
                self.learn_selfplay(episode, learning_rate, game, policy_net, RANDOM_CHOICE, reward_type='explore')
                self.learn_selfplay_black(episode, learning_rate, game, policy_net, RANDOM_CHOICE, reward_type='reversed-explore')
                #loop_code_advanced1
        '''




