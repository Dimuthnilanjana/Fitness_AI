import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score  # Add this import
import joblib
import os
from preprocess import preprocess_data
import logging
import numpy as np
import json
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_clusters(kmeans, scaled_features, feature_names):
    """Analyze cluster characteristics"""
    for i in range(kmeans.n_clusters):
        # Get samples in this cluster
        cluster_samples = scaled_features[kmeans.labels_ == i]
        cluster_center = kmeans.cluster_centers_[i]
        
        # Calculate cluster statistics
        size = len(cluster_samples)
        percentage = (size / len(scaled_features)) * 100
        
        logger.info(f"\nCluster {i} Analysis:")
        logger.info(f"Size: {size} samples ({percentage:.1f}%)")
        logger.info("Cluster Center characteristics:")
        for feat, val in zip(feature_names, cluster_center):
            logger.info(f"- {feat}: {val:.2f}")

def train_model(n_clusters=25):  # Changed from 30 to 25 for optimal performance
    """Train the KMeans model with optimized parameters"""
    try:
        # Get preprocessed features
        scaled_features, scaler, feature_names = preprocess_data(
            'data/bmi.csv',
            'data/mealplans.csv',
            'data/nutrition.csv',
            'data/workouts.csv'
        )

        logger.info("Starting model training with optimal parameters...")

        # Initialize KMeans with optimized configuration
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=20,          # Increased for better stability
            max_iter=300,       # Increased for better convergence
            tol=1e-5,
            init='k-means++',
            verbose=1          # Set to 1 to see convergence progress
        )
        
        # Fit the model
        kmeans.fit(scaled_features)
        
        # Calculate quality metrics
        silhouette = silhouette_score(scaled_features, kmeans.labels_)
        inertia = kmeans.inertia_
        
        # Log detailed metrics
        logger.info("\nModel Quality Metrics:")
        logger.info(f"Number of clusters: {n_clusters}")
        logger.info(f"Silhouette Score: {silhouette:.3f}")
        logger.info(f"Inertia: {inertia:.2f}")
        logger.info(f"Iterations to converge: {kmeans.n_iter_}")
        
        # Save model artifacts
        os.makedirs('models', exist_ok=True)
        joblib.dump(kmeans, 'models/model.pkl')
        joblib.dump(scaler, 'models/scaler.pkl')
        
        # Save detailed cluster analysis
        analyze_and_save_clusters(kmeans, scaled_features, feature_names)
        
        return kmeans, scaler, silhouette
        
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise

def analyze_and_save_clusters(kmeans, scaled_features, feature_names):
    """Analyze clusters and save detailed information"""
    cluster_info = {}
    
    for i in range(kmeans.n_clusters):
        cluster_mask = kmeans.labels_ == i
        cluster_samples = scaled_features[cluster_mask]
        center = kmeans.cluster_centers_[i]
        
        # Calculate additional metrics
        cluster_info[str(i)] = {
            'size': int(cluster_samples.shape[0]),
            'percentage': float((cluster_samples.shape[0] / len(scaled_features)) * 100),
            'center': center.tolist(),
            'focus': get_cluster_focus(center, feature_names),
            'intensity_level': get_intensity_level(center, feature_names),
            'recommended_days': get_recommended_days(center, feature_names),
            'dominant_features': get_dominant_features(center, feature_names),
            'metrics': {
                'std_dev': np.std(cluster_samples, axis=0).tolist(),
                'mean': np.mean(cluster_samples, axis=0).tolist(),
                'median': np.median(cluster_samples, axis=0).tolist(),
                'compactness': float(np.mean(np.linalg.norm(cluster_samples - center, axis=1)))
            },
            'feature_names': feature_names.tolist()
        }
    
    # Save detailed analysis
    os.makedirs('models', exist_ok=True)
    with open('models/cluster_analysis.json', 'w') as f:
        json.dump(cluster_info, f, indent=2)
    
    # Save summary metrics
    summary = {
        'n_clusters': kmeans.n_clusters,
        'silhouette_score': float(silhouette_score(scaled_features, kmeans.labels_)),
        'inertia': float(kmeans.inertia_),
        'n_iterations': int(kmeans.n_iter_),
        'feature_names': feature_names.tolist(),
        'total_samples': len(scaled_features),
        'timestamp': datetime.now().isoformat()
    }
    
    with open('models/model_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

def get_cluster_focus(center, feature_names):
    """Determine cluster focus based on center values"""
    features = dict(zip(feature_names, center))
    
    if features['protein'] > 0.5 and features['intensity'] > 0:
        return 'strength_training'
    elif features['calories'] < 0 and features['intensity'] > 0:
        return 'weight_loss'
    elif features['carbohydrate'] > 0.5 and features['intensity'] > 0:
        return 'endurance'
    else:
        return 'general_fitness'

def get_intensity_level(center, feature_names):
    """Get workout intensity level based on cluster center"""
    features = dict(zip(feature_names, center))
    intensity = features['intensity']
    
    if intensity > 0.5:
        return "High"
    elif intensity > -0.5:
        return "Moderate"
    else:
        return "Low"

def get_recommended_days(center, feature_names):
    """Get recommended workout days based on cluster center"""
    features = dict(zip(feature_names, center))
    fitness_level = features['intensity']
    
    if fitness_level > 0.5:
        return 5
    elif fitness_level > -0.5:
        return 4
    else:
        return 3

def get_dominant_features(center, feature_names):
    """Identify dominant features in cluster center"""
    features = dict(zip(feature_names, center))
    # Get top 3 features by absolute value
    sorted_features = sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)
    return {k: float(v) for k, v in sorted_features[:3]}

# Update main block to use 25 clusters
if __name__ == "__main__":
    kmeans, scaler, silhouette = train_model(n_clusters=25)  # Changed from 30 to 25
    logger.info("\nTraining completed successfully!")
    logger.info(f"Final model saved with {25} clusters")