import Agent
import hex_engine


BOARD_SIZE = 7
NUMBER_OF_EVAL_GAMES = 20

def evaluate(resnet_model, convnet_model):

    model1_random_won = 0
    model2_radnom_won = 0

    game = hex_engine.hexPosition(BOARD_SIZE)

    resnet_agent = Agent.REINFORCE_Agent(board_size=BOARD_SIZE)
    resnet_agent.load_model(path_to_model=resnet_model)
    resnet_agent.evaluate_agent(game=game, baseline_model=False, number_of_games=1000)
    resnet_won = resnet_agent.ratio_won_to_played

    print(f'win_ratio{resnet_won/NUMBER_OF_EVAL_GAMES}')

    resnet_agent2 = Agent.REINFORCE_Agent(board_size=BOARD_SIZE)
    resnet_agent2.load_model(path_to_model=convnet_model)
    resnet_agent2.evaluate_agent(game=game, baseline_model=False, number_of_games=1000)
    model1_random_won22 = resnet_agent.ratio_won_to_played
    print(f'win_ratio{resnet_won/NUMBER_OF_EVAL_GAMES}')

if __name__ == '__main__':

    resnet_model = './7x7_ResNet50_4dd153f6-30e3-11ef-93aa-3e22fb4c01bd_ResNet50.keras'
    convnet_model = './7x7_Convnet_9ca6ded8-31b0-11ef-bb9f-3e22fb4c01bd_conv_net_.keras'

    game = hex_engine.hexPosition(BOARD_SIZE)

    resnet_agent = Agent.REINFORCE_Agent(board_size=BOARD_SIZE)
    resnet_agent.load_model_eval(path_to_model=resnet_model)

    resnet_agent.evaluate_agent(another_player_path=convnet_model, game=game, baseline_model=False, number_of_games=NUMBER_OF_EVAL_GAMES)
    resnet_won = resnet_agent.eval_random_won
    print(f'ResNET50 - Win Ratio{resnet_won/NUMBER_OF_EVAL_GAMES}')

    covnet_agent = Agent.REINFORCE_Agent(board_size=BOARD_SIZE)
    covnet_agent.load_model_eval(path_to_model=convnet_model)
    covnet_agent.evaluate_agent(another_player_path=convnet_model, game=game, baseline_model=False, number_of_games=NUMBER_OF_EVAL_GAMES)
    convnet_won = covnet_agent.eval_random_won

    print(f'ResNET50 - Win Ratio: {resnet_won/NUMBER_OF_EVAL_GAMES}')
    print(f'ConvNet - Win Ratio: {convnet_won/NUMBER_OF_EVAL_GAMES}')
    