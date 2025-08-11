from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam

def build_dqn_model(state_size: int, action_size: int, learning_rate: float = 0.001) -> Sequential:
    """Build a deeper neural network for DQN"""
    model = Sequential([
        Input(shape=(state_size,)),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(action_size, activation='linear')
    ])
    
    model.compile(
        loss='mse',
        optimizer=Adam(learning_rate=learning_rate)
    )
    
    return model