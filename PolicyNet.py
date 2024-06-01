import tensorflow as tf
from keras import layers, models, initializers, Input
import keras



def create_hex_policy_net(board_size):
    '''
    This is the policy net, i.e. it returns the probabilities of actions to be taken. 
    Input Shape: First two specify the field on the board, third specifies if the field is empty, occuped by player or oponent: 0 ,1, -1
    '''

    
    initializer = initializers.HeNormal()


    input_layer = Input(shape=(board_size, board_size, 1))
    

    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer=initializer)(input_layer)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=initializer)(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=initializer)(x)
    
  
    x = layers.Flatten()(x)
    
   
    x = layers.Dense(256, activation='relu', kernel_initializer=initializer)(x)
    x = layers.Dense(128, activation='relu', kernel_initializer=initializer)(x)
 
    output_layer = layers.Dense(board_size * board_size, activation='softmax', kernel_initializer=initializer)(x)
    

    model = models.Model(inputs=input_layer, outputs=output_layer)
    model.name = "hex_policy_conv_net_"
    
    return model
    
def create_complex_hex_policy_net(board_size):
    
    initializer = initializers.HeNormal()
    input_layer = layers.Input(shape=(board_size, board_size, 1))

    # Convolutional layers with Batch Normalization and ReLU activation
    x = layers.Conv2D(32, (3, 3), activation=None, padding='same', kernel_initializer=initializer)(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(64, (3, 3), activation=None, padding='same', kernel_initializer=initializer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(128, (3, 3), activation=None, padding='same', kernel_initializer=initializer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Optional: Add more convolutional layers if needed
    x = layers.Conv2D(256, (3, 3), activation=None, padding='same', kernel_initializer=initializer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Global Average Pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Fully connected layers with Dropout
    x = layers.Dense(256, activation='relu', kernel_initializer=initializer)(x)
    x = layers.Dropout(0.5)(x)  # Dropout rate of 0.5

    x = layers.Dense(128, activation='relu', kernel_initializer=initializer)(x)
    x = layers.Dropout(0.5)(x)  # Dropout rate of 0.5

    # Output layer
    output_layer = layers.Dense(board_size * board_size, activation='softmax', kernel_initializer=initializer)(x)

    # Create model
    model = models.Model(inputs=input_layer, outputs=output_layer)
    model.name = "complex_hex_policy_conv_net_"
    
    return model


    
    
def create_simple_policy_net(board_size):
  
        initializer = initializers.HeNormal()
        

        input_layer = Input(shape=(board_size * board_size, 1))
        
     
        x = layers.Dense(256, activation='relu', kernel_initializer=initializer)(input_layer)
        x = layers.Dense(128, activation='relu', kernel_initializer=initializer)(x)
        
      
        output_layer = layers.Dense(board_size * board_size, activation='softmax', kernel_initializer=initializer)(x)
        
        model = models.Model(inputs=input_layer, outputs=output_layer)
        
        model.name = "simple_hex_policy_net_"
        
        return model

