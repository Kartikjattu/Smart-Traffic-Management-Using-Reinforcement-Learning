import numpy as np
from gym import Env, spaces
from typing import Tuple, Dict
from environment.data_loader import TrafficDataLoader

class TrafficEnvironment(Env):
    def __init__(self, data_path: str, location_id: str = None):
        super(TrafficEnvironment, self).__init__()
        self.data_loader = TrafficDataLoader(data_path)
        
        # Use specified location or first available
        self.location_id = location_id or self.data_loader.get_all_locations()[0]
        self.location_data = self.data_loader.get_location_data(self.location_id)
        
        # Define action space (switch to different light phase)
        self.action_space = spaces.Discrete(4)  # 4 light phases
        
        # Define observation space
        self.observation_space = spaces.Box(
            low=0,
            high=np.inf,
            shape=(len(self.data_loader.get_feature_columns()),),
            dtype=np.float32
        )
        
        # Initialize state
        self.current_step = 0
        self.max_steps = len(self.location_data) - 1
        
    def _get_state(self) -> np.ndarray:
        """Convert current data row to state vector"""
        row = self.location_data.iloc[self.current_step]
        features = []
        
        # Numerical features
        features.append(row['vehicle_count'])
        features.append(row['avg_speed_kmh'])
        
        # Categorical features (encoded)
        congestion_map = {'Light': 0, 'Moderate': 1, 'Heavy': 2, 'Severe': 3}
        features.append(congestion_map.get(row['congestion_level'], 0))
        
        incident_map = {'No': 0, 'Yes (Breakdown)': 1, 'Yes (Accident)': 2, 'Yes (Cleared)': 3}
        features.append(incident_map.get(row['incident_reported'], 0))
        
        weather_map = {'Clear': 0, 'Rain': 1, 'Snow': 2, 'Fog': 3}
        features.append(weather_map.get(row['weather_condition'], 0))
        
        # Vehicle distribution
        vehicle_dist = row['vehicle_type_distribution']
        features.append(vehicle_dist.get('cars', 0))
        features.append(vehicle_dist.get('trucks', 0))
        features.append(vehicle_dist.get('bikes', 0))
        features.append(vehicle_dist.get('buses', 0))
        
        # Pedestrian and transit
        features.append(row['pedestrian_count'])
        features.append(row['public_transit_count'])
        
        return np.array(features, dtype=np.float32)
    
    def reset(self) -> np.ndarray:
        """Reset the environment"""
        self.current_step = 0
        return self._get_state()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one time step in the environment
        Returns:
            tuple: (observation, reward, done, info)
        """
        if self.current_step >= self.max_steps:
            raise ValueError("Episode has already ended.")
            
        # Get current and next state data
        current_row = self.location_data.iloc[self.current_step]
        next_row = self.location_data.iloc[self.current_step + 1]
        
        # Calculate reward based on action and traffic flow
        reward = self._calculate_reward(current_row, next_row, action)
        
        # Move to next time step
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        # Get next state
        next_state = self._get_state()
        
        # Additional info
        info = {
            'timestamp': current_row['timestamp'],
            'location': current_row['location_name'],
            'step': self.current_step,
            'signal_changed': action != current_row['signal_status']
        }
        
        return next_state, reward, done, info
    
    def _calculate_reward(self, current_row, next_row, action) -> float:
        """Calculate reward based on traffic conditions"""
        reward = 0
        
        # Positive reward for reducing congestion
        congestion_map = {'Light': 0, 'Moderate': -1, 'Heavy': -3, 'Severe': -5}
        reward += congestion_map.get(next_row['congestion_level'], 0)
        
        # Positive reward for maintaining flow when light is green
        if action == 0:  # Assuming 0 is green
            reward += next_row['avg_speed_kmh'] / 10
            
        # Penalty for changing lights too frequently
        if action != current_row['signal_status']:
            reward -= 0.5
            
        # Large penalty for incidents
        if 'Yes' in next_row['incident_reported']:
            reward -= 10
            
        return reward
    
    def render(self, mode='human'):
        """Render the current state"""
        if mode == 'human':
            row = self.location_data.iloc[self.current_step]
            print(f"\nStep: {self.current_step}")
            print(f"Time: {row['timestamp']}")
            print(f"Location: {row['location_name']}")
            print(f"Vehicles: {row['vehicle_count']}")
            print(f"Avg Speed: {row['avg_speed_kmh']} km/h")
            print(f"Congestion: {row['congestion_level']}")
            print(f"Signal: {row['signal_status']}")