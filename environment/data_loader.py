import pandas as pd
import ast
from typing import Dict, List

class TrafficDataLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = self._load_data()
        
    def _load_data(self) -> pd.DataFrame:
        """Load and preprocess the CSV data"""
        df = pd.read_csv(self.file_path)
        
        # Convert string dict to actual dict
        df['vehicle_type_distribution'] = df['vehicle_type_distribution'].apply(ast.literal_eval)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by location and timestamp
        df = df.sort_values(['location_id', 'timestamp'])
        
        return df
    
    def get_location_data(self, location_id: str) -> pd.DataFrame:
        """Get data for a specific location"""
        return self.data[self.data['location_id'] == location_id].copy()
    
    def get_all_locations(self) -> List[str]:
        """Get list of all unique location IDs"""
        return self.data['location_id'].unique().tolist()
    
    def get_feature_columns(self) -> List[str]:
        """Get list of feature columns to use in model"""
        return [
            'vehicle_count',
            'avg_speed_kmh',
            'congestion_level',
            'incident_reported',
            'weather_condition',
            'vehicle_type_distribution',
            'pedestrian_count',
            'public_transit_count'
        ]