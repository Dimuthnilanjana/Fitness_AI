import pytest
import json
import os
import numpy as np
from src.generate_plan import generate_plan

# Create test data directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Enhanced test cluster data matching ML model features
TEST_CLUSTER_DATA = {
    "0": {
        "focus": "weight_loss",
        "intensity_level": "moderate",
        "recommended_days": 4,
        "cardio_ratio": 0.5,
        "strength_ratio": 0.3,
        "flexibility_ratio": 0.2,
        "daily_calories": 2000,
        "protein_ratio": 0.3,
        "carb_ratio": 0.4,
        "fat_ratio": 0.3,
        "meal_frequency": 4,
        "center": [70, 1.7, 30, 24.2, 4, 7, 2000, 0.3, 0.4, 0.3, 25, 0, 1, 0]
    }
}

def setup_module():
    """Setup test environment with ML model test data"""
    with open('models/cluster_analysis.json', 'w') as f:
        json.dump(TEST_CLUSTER_DATA, f)

def test_generate_plan():
    """Test ML-based generation of 30-day workout and meal plans"""
    cluster = 0
    user_data = {
        'weight': 70,
        'height': 170,
        'age': 30,
        'equipment': 'bodyweight',
        'fitness_level': 'Intermediate',
        'days_per_week': 4,
        'sleep_hours': 7,
        'calorie_target': 2000,
        'diet_type': 'balanced',
        'intensity': 0,
        'exercise_type': 1,
        'protein': 0.3,
        'carbohydrate': 0.4,
        'total_fat': 0.3,
        'meals_per_day': 4
    }
    
    plan = generate_plan(cluster, user_data)
    
    # Validate ML model integration
    assert isinstance(plan, dict)
    assert all(key in plan for key in [
        'cluster', 'focus', 'intensity_level', 'recommended_days',
        'workout_plan', 'nutrition_plan'
    ])
    
    # Validate 30-day workout plan
    workout_plan = plan['workout_plan']
    assert len(workout_plan) == 30, "Should generate full 30 day plan"
    
    # Check workout distribution matches ML model prediction
    workout_types = [day['type'] for day in workout_plan.values() if day['type'] != 'Rest']
    cardio_days = sum(1 for t in workout_types if t == 'Cardio')
    strength_days = sum(1 for t in workout_types if t == 'Strength')
    flexibility_days = sum(1 for t in workout_types if t == 'Flexibility')
    
    # Verify ratios match cluster data (allowing 10% deviation)
    total_workout_days = len([d for d in workout_plan.values() if d['type'] != 'Rest'])
    assert abs(cardio_days/total_workout_days - TEST_CLUSTER_DATA["0"]["cardio_ratio"]) < 0.1
    assert abs(strength_days/total_workout_days - TEST_CLUSTER_DATA["0"]["strength_ratio"]) < 0.1
    
    # Validate 30-day meal plan
    nutrition_plan = plan['nutrition_plan']
    assert 'daily_targets' in nutrition_plan
    assert all(key in nutrition_plan['daily_targets'] for key in [
        'calories', 'protein', 'carbs', 'fat', 'fiber'
    ])
    
    # Verify meal plan matches ML model predictions
    daily_targets = nutrition_plan['daily_targets']
    assert abs(daily_targets['calories'] - TEST_CLUSTER_DATA["0"]["daily_calories"]) < 100
    assert abs(daily_targets['protein']/daily_targets['calories']*4 - 
              TEST_CLUSTER_DATA["0"]["protein_ratio"]) < 0.05
    
    # Validate meal distribution
    assert 'meals' in nutrition_plan
    assert len(nutrition_plan['meals']) == user_data['meals_per_day']
    
    # Verify each day has proper exercise details
    for day, workout in workout_plan.items():
        if workout['type'] != 'Rest':
            assert 'exercises' in workout
            assert len(workout['exercises']) > 0
            assert 'intensity' in workout
            assert isinstance(workout['exercises'], list)

def teardown_module():
    """Cleanup test data"""
    if os.path.exists('models/cluster_analysis.json'):
        os.remove('models/cluster_analysis.json')

#python -m pytest tests/test_generate_plan.py -v