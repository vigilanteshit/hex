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
    model.name = "simple_conv_net_"
    
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
    model.name = "conv_net_"
    
    return model


    
    
def create_simple_policy_net(board_size):
  
        initializer = initializers.HeNormal()
        

        input_layer = Input(shape=(board_size, board_size, 1))
        
     
        x = layers.Dense(256, activation='relu', kernel_initializer=initializer)(input_layer)
        x = layers.Dense(128, activation='relu', kernel_initializer=initializer)(x)
        
      
        output_layer = layers.Dense(board_size * board_size, activation='softmax', kernel_initializer=initializer)(x)
        
        model = models.Model(inputs=input_layer, outputs=output_layer)
        
        model.name = "simple_dense_net_"
        
        return model


def create_dense_nn_policy_net(board_size):
  
        initializer = initializers.HeNormal()
        

        input_layer = Input(shape=(board_size, board_size, 1))
        
        x = layers.Flatten()(input_layer)
        
        x = layers.Dense(512, activation='relu', kernel_initializer=initializer)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(256, activation='relu', kernel_initializer=initializer)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(128, activation='relu', kernel_initializer=initializer)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(64, activation='relu', kernel_initializer=initializer)(x)
        
        output_layer = layers.Dense(board_size * board_size, activation='softmax', kernel_initializer=initializer)(x)
        
        model = models.Model(inputs=input_layer, outputs=output_layer)
        
        model.name = "dense_nn_net_"
        
        return model





def identity_block(X, f, filters, stage, block):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    F1, F2, F3 = filters

    X_shortcut = X

    X = layers.Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a')(X)
    X = layers.BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = layers.Activation('relu')(X)

    X = layers.Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b')(X)
    X = layers.BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = layers.Activation('relu')(X)

    X = layers.Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c')(X)
    X = layers.BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    X = layers.Add()([X, X_shortcut])
    X = layers.Activation('relu')(X)

    return X

def convolutional_block(X, f, filters, stage, block, s=2):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    F1, F2, F3 = filters

    X_shortcut = X

    X = layers.Conv2D(F1, (1, 1), strides=(s, s), name=conv_name_base + '2a')(X)
    X = layers.BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = layers.Activation('relu')(X)

    X = layers.Conv2D(F2, (f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b')(X)
    X = layers.BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = layers.Activation('relu')(X)

    X = layers.Conv2D(F3, (1, 1), strides=(1, 1), name=conv_name_base + '2c')(X)
    X = layers.BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    X_shortcut = layers.Conv2D(F3, (1, 1), strides=(s, s), name=conv_name_base + '1')(X_shortcut)
    X_shortcut = layers.BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    X = layers.Add()([X, X_shortcut])
    X = layers.Activation('relu')(X)

    return X

def ResNet50(input_shape=(64, 64, 1), classes=64*64):
    X_input = Input(input_shape)

    # Adjusting the initial convolution layer for smaller input sizes
    X = layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='conv1')(X_input)
    X = layers.BatchNormalization(axis=3, name='bn_conv1')(X)
    X = layers.Activation('relu')(X)

    # Only apply MaxPooling if the input size is large enough
    if input_shape[0] > 3 and input_shape[1] > 3:
        X = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same')(X)

    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    X = layers.GlobalAveragePooling2D()(X)
    X = layers.Dense(classes, activation='softmax', name='fc' + str(classes))(X)

    model = models.Model(inputs=X_input, outputs=X, name='ResNet50')

    return model


