import numpy as np
from environment.traffic_env import TrafficEnvironment
from agents.dqn_agent_pytorch import DQNAgent  # Changed from dqn_agent to dqn_agent_pytorch
from utils.logger import setup_logger

def simulate(data_path: str = 'data/traffic-management.csv',
             model_path: str = 'data/trained_models/dqn_traffic.pth'):  # Changed from .h5 to .pth
    """Run simulation with trained model"""
    
    # Setup logger
    logger = setup_logger()
    
    # Create environment
    env = TrafficEnvironment(data_path)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Create agent and load weights
    agent = DQNAgent(state_size, action_size)
    agent.load(model_path)
    agent.epsilon = 0.01  # Minimal exploration
    
    # Run simulation
    state = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        env.render()
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        
        logger.info(f"Action: {action}, Reward: {reward:.2f}, Next Congestion: {env.location_data.iloc[env.current_step]['congestion_level']}")
        
        state = next_state
        total_reward += reward
    
    logger.info(f"Simulation complete. Total reward: {total_reward:.2f}")

if __name__ == "__main__":
    simulate()