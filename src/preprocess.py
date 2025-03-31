from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import re
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(bmi_path, meals_path, nutrition_path, workouts_path):
    """Load all required datasets"""
    try:
        bmi_df = pd.read_csv(bmi_path)
        meals_df = pd.read_csv(meals_path)
        nutrition_df = pd.read_csv(nutrition_path)
        workouts_df = pd.read_csv(workouts_path)
        
        logger.info("Successfully loaded all datasets")
        return bmi_df, meals_df, nutrition_df, workouts_df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def clean_bmi_data(bmi_df):
    """Clean and process BMI data"""
    bmi_df.dropna(inplace=True)
    bmi_df['Bmi'] = bmi_df['Weight'] / (bmi_df['Height'] ** 2)
    bmi_df['BmiClass'] = pd.cut(
        bmi_df['Bmi'], 
        bins=[0, 18.5, 24.9, 29.9, 34.9, 39.9, np.inf],
        labels=['Underweight', 'Normal', 'Overweight', 'Obese1', 'Obese2', 'Obese3']
    )
    
    # Add exercise frequency and sleep hours
    bmi_df['Days_Per_Week'] = 3
    bmi_df['Sleep_Hours'] = 7
    
    return bmi_df

def clean_workouts_data(workouts_df):
    """Clean and process workouts data"""
    # Create a copy to avoid chained assignment warning
    df = workouts_df.copy()
    
    # Convert level to numeric
    level_map = {'Beginner': 1, 'Intermediate': 2, 'Advanced': 3}
    df['Level_Numeric'] = df['Level'].map(level_map).fillna(1)
    
    # Convert type to categorical
    df['Type_Code'] = pd.Categorical(df['Type']).codes
    
    # Clean equipment field properly
    df['Equipment'] = df['Equipment'].fillna('bodyweight')
    
    # Add duration field
    df['Duration'] = 30  # Default 30 minutes
    
    # Add rating conversion
    df['Rating_Numeric'] = pd.to_numeric(df['Rating'], errors='coerce').fillna(3)
    
    return df

def extract_numeric(value):
    """Extract numeric values from strings"""
    if isinstance(value, (int, float)):
        return float(value)
    
    if isinstance(value, str):
        # Extract numbers from string (including decimals)
        matches = re.findall(r'[-+]?\d*\.\d+|\d+', value)
        if matches:
            return float(matches[0])
    
    return 0.0  # Default value if no number found

def normalize_nutrition_data(nutrition_df):
    """Clean and normalize nutrition data"""
    columns_to_normalize = [
        'calories', 'total_fat', 'cholesterol', 'sodium', 
        'fiber', 'protein', 'carbohydrate'
    ]
    
    for col in columns_to_normalize:
        if (col in nutrition_df.columns):
            nutrition_df[col] = nutrition_df[col].apply(extract_numeric)
        else:
            logger.warning(f"Column {col} not found in nutrition data")
            nutrition_df[col] = 0
            
    return nutrition_df

def combine_features(bmi_df, nutrition_df, workouts_df):
    """Combine all features for clustering with proper alignment"""
    
    # Reset indices to ensure proper alignment
    bmi_df = bmi_df.reset_index(drop=True)
    nutrition_df = nutrition_df.reset_index(drop=True)
    workouts_df = workouts_df.reset_index(drop=True)
    
    # Create base size - use smallest dataset size to avoid data misalignment
    min_size = min(len(bmi_df), len(nutrition_df), len(workouts_df))
    logger.info(f"Using minimum dataset size: {min_size}")
    
    features = pd.DataFrame({
        # BMI and physical features - take only min_size rows
        'weight': bmi_df['Weight'].astype(float)[:min_size],
        'height': bmi_df['Height'].astype(float)[:min_size],
        'age': bmi_df['Age'].astype(float)[:min_size],
        'bmi': bmi_df['Bmi'].astype(float)[:min_size],
        'days_per_week': bmi_df['Days_Per_Week'].astype(float)[:min_size],
        'sleep_hours': bmi_df['Sleep_Hours'].astype(float)[:min_size],
        
        # Nutrition features - ensure numeric and align size
        'calories': nutrition_df['calories'].astype(float)[:min_size],
        'protein': nutrition_df['protein'].astype(float)[:min_size],
        'carbohydrate': nutrition_df['carbohydrate'].astype(float)[:min_size],
        'total_fat': nutrition_df['total_fat'].astype(float)[:min_size],
        'fiber': nutrition_df['fiber'].astype(float)[:min_size],
        
        # Workout features - align size
        'intensity': workouts_df['Level_Numeric'].astype(float)[:min_size],
        'exercise_type': workouts_df['Type_Code'].astype(float)[:min_size],
        'rating': workouts_df['Rating_Numeric'].astype(float)[:min_size]
    })
    
    # Remove any remaining NaN values
    features = features.fillna(features.mean())
    
    logger.info(f"Combined feature shape: {features.shape}")
    return features

def preprocess_data(bmi_path, meals_path, nutrition_path, workouts_path):
    """Main preprocessing function"""
    try:
        # Load data
        bmi_df, meals_df, nutrition_df, workouts_df = load_data(
            bmi_path, meals_path, nutrition_path, workouts_path
        )
        
        logger.info("Data shapes before preprocessing:")
        logger.info(f"BMI data: {bmi_df.shape}")
        logger.info(f"Nutrition data: {nutrition_df.shape}")
        logger.info(f"Workouts data: {workouts_df.shape}")
        
        # Clean individual datasets
        bmi_df = clean_bmi_data(bmi_df)
        nutrition_df = normalize_nutrition_data(nutrition_df)
        workouts_df = clean_workouts_data(workouts_df)
        
        # Combine features
        features = combine_features(bmi_df, nutrition_df, workouts_df)
        
        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        logger.info("\nPreprocessing completed:")
        logger.info(f"Final feature shape: {scaled_features.shape}")
        logger.info(f"Features included: {features.columns.tolist()}")
        
        return scaled_features, scaler, features.columns
        
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        raise

if __name__ == "__main__":
    # Test preprocessing
    scaled_features, scaler, feature_names = preprocess_data(
        'data/bmi.csv',
        'data/mealplans.csv',
        'data/nutrition.csv',
        'data/workouts.csv'
    )
    print("Feature names:", feature_names)
    print("Scaled features shape:", scaled_features.shape)