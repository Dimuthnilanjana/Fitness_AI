import json
from src.app import generate_workout_plan, generate_nutrition_plan

def generate_plan(cluster, user_data):
    """Generate workout and nutrition plan based on cluster and user data"""
    try:
        # Load cluster information
        with open('models/cluster_analysis.json', 'r') as f:
            cluster_info = json.load(f)
        
        cluster_data = cluster_info[str(cluster)]
        
        # Add cluster info to user data
        user_data.update({
            'cluster': cluster,
            'intensity_level': cluster_data['intensity_level'],
            'recommended_days': cluster_data['recommended_days']
        })
        
        return {
            'cluster': cluster,
            'focus': cluster_data['focus'],
            'intensity_level': cluster_data['intensity_level'],
            'recommended_days': cluster_data['recommended_days'],
            'workout_plan': generate_workout_plan(user_data),  # Pass single argument
            'nutrition_plan': generate_nutrition_plan(cluster_data, user_data)
        }
    except Exception as e:
        raise ValueError(f"Error generating plan: {str(e)}")