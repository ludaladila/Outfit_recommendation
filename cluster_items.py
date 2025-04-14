'''use kmeans to cluster all images based on embedding , output as image_clusters.csv '''
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

def load_embeddings(feature_csv: str):
    df = pd.read_csv(feature_csv)
    df["filename"] = df["image_path"].apply(lambda x: os.path.basename(x))
    return df

def run_kmeans(X: np.ndarray, n_clusters: int = 6):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(X)
    return labels, model

def plot_clusters(X: np.ndarray, labels: np.ndarray, title="KMeans Clusters"):
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='tab10')
    plt.title(title)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.colorbar(scatter, ticks=range(len(set(labels))))
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    csv_folder = "csv"
    os.makedirs(csv_folder, exist_ok=True)

    feature_csv = os.path.join(csv_folder, "image_features.csv")
    output_csv = os.path.join(csv_folder, "image_clusters.csv")
    n_clusters = 3  

    # Load features
    df = load_embeddings(feature_csv)
    X = df.drop(columns=["image_path", "filename"]).values

    # Run clustering
    labels, model = run_kmeans(X, n_clusters=n_clusters)
    df["cluster"] = labels

    # Save results
    df.to_csv(output_csv, index=False)
    print(f"Saved cluster results to: {output_csv}")

    # Optional: visualize
    plot_clusters(X, labels)
