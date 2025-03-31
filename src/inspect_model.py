import joblib
import numpy as np

def inspect_kmeans_model():
    try:
        # Load the model
        model = joblib.load('models/model.pkl')
        
        # Get number of clusters (fix attribute name)
        n_clusters = model.n_clusters
        
        # Get cluster centers
        centers = model.cluster_centers_
        
        print(f"Model type: {type(model)}")
        print(f"Number of clusters: {n_clusters}")
        print("\nCluster centers:")
        for i, center in enumerate(centers):
            print(f"\nCluster {i}:")
            feature_names = ['weight', 'height', 'age', 'days_per_week', 'sleep_hours', 'dummy']
            for name, value in zip(feature_names, center):
                print(f"{name}: {value:.2f}")
        
        return n_clusters
    
    except Exception as e:
        print(f"Error inspecting model: {str(e)}")
        return None

if __name__ == "__main__":
    inspect_kmeans_model()