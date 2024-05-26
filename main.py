import hex_engine
import PolicyNet
import Agent
import tensorflow as tf
import matplotlib.pyplot as plt

NUMBER_OF_EPISODES = 1000
LEARNING_RATE = 1e-3
BOARD_SIZE = 7

if __name__ == '__main__':
    game = hex_engine.hexPosition(BOARD_SIZE)

    policy_net = PolicyNet.create_hex_policy_net(BOARD_SIZE)
    
    print(policy_net.summary())
    agent = Agent.REINFORCE_Agent(BOARD_SIZE)
    board_tensor = tf.convert_to_tensor(game.board, dtype=tf.int32)
    game_state_tensor = tf.reshape(board_tensor, (BOARD_SIZE, BOARD_SIZE, 1))
    
    played, won, ratio = agent.learn_selfplay(NUMBER_OF_EPISODES, LEARNING_RATE, game)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(ratio)), ratio, marker='o', markersize=2)
    plt.xlabel('Games Played')
    plt.ylabel('Ratio: Won/Played')
    plt.title('Ratio of Won Games to Played Games')
    plt.grid(True)
    plt.show()
    
    #played, won, ratio = agent.learn_random(NUMBER_OF_EPISODES, LEARNING_RATE, game)
    #predecessor_model = agent.load_model("./hex_policy_conv_net_2024-05-26 20:23:55.keras")
    #played, won, ratio = agent.learn_from_other_model(NUMBER_OF_EPISODES, LEARNING_RATE, game, predecessor_model)
    