from environment.traffic_env import TrafficEnvironment
from agents.dqn_agent_pytorch import DQNAgent
from utils.logger import setup_logger
import os

def train(episodes=1000,
          data_path='data/traffic-management.csv',
          model_save_path='data/trained_models/dqn_traffic.pth',
          batch_size=32):
    
    # Setup logger
    logger = setup_logger()
    
    # Create environment and agent
    env = TrafficEnvironment(data_path)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    
    # Create directory for model if it doesn't exist
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # Training loop
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            # Train on batch of experiences
            agent.replay(batch_size)
            
        logger.info(f"Episode: {episode + 1}/{episodes}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
        
        # Save model every 100 episodes
        if (episode + 1) % 100 == 0:
            agent.save(model_save_path)
            logger.info(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    train()