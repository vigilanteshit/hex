# hex

### Environment
first try on existing enviornment

mini hex
https://github.com/FirefoxMetzger/minihex
after final decision on algorithmn implement enviornment

implement CNN
https://github.com/harbecke/HexHex

Monte Carlo Tree Search (MCTS)
to simulate game moves and explore the state space efficiently
balance exploration and exploitation using the Upper Confidence bounds for Trees (UCT) algorithm

Self-Play Data Generation
https://github.com/jseppanen/azalea
collect training data from self-play-games, including states, actions, and outcomes


general sources
https://spinningup.openai.com/en/latest/spinningup/rl_intro.html


## Reinforce Agent - Instructions

### main.py
It contains example code on how to execute a simple training run by executing a training method or how to execute parameter tuning with a training method

### play_against_model.py
This script contains examples how to play human2machine or machine2machine

### PolicyNet.py
This modul contains possible policy nets for the agent

### hex_engine
This modul contains the hex game

### Agent
This modul contains the Agent class. It contains all methods to learn and play. The train method lets one train models either with default values of the reward parameters or custom settings. Each
training run creates a training run directory, where a graph of the training and the model is stored. The model is further stored in the model dump. The model dump serves as a model repository for
training: If the training method contains a learn_from_other_model method the model can either be trained on its predecessor model or some random model from the repository. Once the training is finished
the training stats are written in the training log. 

### GridSearch
The GridSearch module alows the training of multiple agents in parallel (the degree of concurrency is dependent on the number_of_jobs parameter). To this end for every parameter a list is passed to the run_gridsearch_job method (note: even a set of only one parameter has to be passed as a list). For every available parameter configuration seperate agents are trained. Besides storing the model in the model dump, every model gets its 
own trainin run directory, where the model and a plot of the training run is stored. Once a process finishes the training of an agent it sends the trainin stats via a queue to the gridsearch_job_listener which writes the data to the training log (therefore the total number of processes occupuied is number_of_jobs + 1). 

### Hyperparameters
Several hyperparameters are available. The parameters are of two different kinds: Hyperparamters in the narrow sense, like learning rate, and reward parameters, which are can be used to shape the reward behavior
environment. 

#### Hyperparamters
- learning_rate: The learning rate of the algorithm
- gradient_clipping_parameter: Used to control large gradients
- number_of_episodes: The number of episodes to train per training layer

#### Reward Parameters
- similarity_panelty: A small negative parameter. It penalizes consecutive moves made along the horizontal axis (white) and encourages the agent to explore the board also vertically
- similarity_penalty_decay_rate: This controls the influence of similarity_panelty parameter in the course of the game. For high values of the parameter, similarity_panelty gets less important later in the game
- random_choice: Controls the probability by which the oponent makes random moves during training
- number_of_eval_games: Controls how many games are played for evaluation (technically not really a hyperparameter)
