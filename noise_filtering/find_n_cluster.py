# Limit OpenBLAS thread count
import os
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'

import random
from pathlib import Path
import click
import numpy as np
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans, GaussianMixture
from pyspark.ml.evaluation import ClusteringEvaluator
from clustering_spark import perform_kmeans_clustering

def load_features(spark, cache_dir, clustering_result_path=None, target_clusters=None):
    """Load cached features as a Spark DataFrame"""
    features_parquet = Path(cache_dir) / "extracted_features.parquet"
    if features_parquet.exists():
        # Read Parquet file directly as a Spark DataFrame
        df = spark.read.parquet(str(features_parquet))
        
        # Filter by clustering result if provided
        if clustering_result_path and target_clusters is not None:
            clustering_result = spark.read.parquet(str(clustering_result_path))
            if 'prediction' in df.columns:
                df = df.drop('prediction')
            df = df.join(clustering_result.select("segmentid", "prediction"), on="segmentid", how="inner")
            df = df.filter(df.prediction.isin(target_clusters))
            df = df.drop('prediction')
            print(f"Filtered to clusters {target_clusters}: {df.count()} records")
        
        # Extract feature columns (excluding segmentid)
        feature_cols = [col for col in df.columns if col.startswith('feature_')]
        feature_cols.sort()  # Sort in order: feature_0, feature_1, ...
        
        print(f"Loaded {df.count()} features")
        print(f"Feature columns: {feature_cols}")
        return df, feature_cols
    else:
        print(f"No cached features found at {features_parquet}")
        return None, None


def find_elbow_point(inertias, k_range):
    """Find the elbow point"""
    # Find the point of steepest change using the second derivative
    if len(inertias) < 3:
        return k_range[1] if len(k_range) > 1 else k_range[0]
    
    # Compute second derivative
    second_derivatives = []
    for i in range(1, len(inertias) - 1):
        second_deriv = inertias[i+1] - 2*inertias[i] + inertias[i-1]
        second_derivatives.append(second_deriv)
    
    # Find the index of the largest second derivative value
    max_second_deriv_idx = np.argmax(second_derivatives)
    elbow_k = k_range[max_second_deriv_idx + 1]  # +1 because we start from index 1
    
    return elbow_k


def plot_elbow_curve(k_range, inertias, silhouette_scores, output_dir):
    """Plot and save the elbow curve"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Inertia (Within-cluster sum of squares) plot
    ax1.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow Method for Optimal k')
    ax1.grid(True, alpha=0.3)
    
    # Mark the elbow point
    elbow_k = find_elbow_point(inertias, k_range)
    elbow_idx = k_range.index(elbow_k)
    ax1.axvline(x=elbow_k, color='red', linestyle='--', alpha=0.7, 
                label=f'Elbow point: k={elbow_k}')
    ax1.legend()
    
    # Silhouette Score plot
    ax2.plot(k_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Score vs Number of Clusters')
    ax2.grid(True, alpha=0.3)
    
    # Mark the best silhouette score point
    best_silhouette_idx = np.argmax(silhouette_scores)
    best_silhouette_k = k_range[best_silhouette_idx]
    ax2.axvline(x=best_silhouette_k, color='red', linestyle='--', alpha=0.7,
                label=f'Best silhouette: k={best_silhouette_k}')
    ax2.legend()
    
    plt.tight_layout()
    
    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path / "elbow_curve.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return elbow_k, best_silhouette_k


def find_optimal_clusters(spark, df, k_range, method='kmeans', output_dir="./outputs"):
    """Find the optimal number of clusters"""
    print(f"Finding optimal number of clusters using {method}...")
    print(f"Testing k values: {k_range}")
    
    inertias = []
    silhouette_scores = []
    
    for k in k_range:
        print(f"Testing k={k}...")
        
        if method == 'kmeans':
            kmeans = KMeans(featuresCol="features", k=k, seed=42)
            model = kmeans.fit(df)
            predictions = model.transform(df)
            inertia = model.summary.trainingCost
        elif method == 'gmm':
            gmm = GaussianMixture(featuresCol="features", k=k, seed=42)
            model = gmm.fit(df)
            predictions = model.transform(df)
            # Use negative log likelihood instead of inertia for GMM
            inertia = -model.summary.logLikelihood
        
        # Compute silhouette score
        evaluator = ClusteringEvaluator()
        silhouette = evaluator.evaluate(predictions)
        
        inertias.append(inertia)
        silhouette_scores.append(silhouette)
        
        print(f"  k={k}: Inertia={inertia:.4f}, Silhouette={silhouette:.4f}")
    
    # Plot elbow curve and find optimal k
    elbow_k, best_silhouette_k = plot_elbow_curve(k_range, inertias, silhouette_scores, output_dir)
    
    print(f"\nResults:")
    print(f"Elbow method suggests k={elbow_k}")
    print(f"Best silhouette score at k={best_silhouette_k}")
    
    # Save results as CSV
    results_df = spark.createDataFrame([
        (k, inertia, silhouette) for k, inertia, silhouette in zip(k_range, inertias, silhouette_scores)
    ], ["k", "inertia", "silhouette_score"])
    
    output_path = Path(output_dir)
    results_df.toPandas().to_csv(output_path / f"{method}_clustering_analysis.csv", index=False)
    
    return elbow_k, best_silhouette_k, inertias, silhouette_scores



@click.command()
@click.option('--cache-dir', type=str, default='./features')
@click.option('--k-min', type=int, default=2, help='Minimum number of clusters to test')
@click.option('--k-max', type=int, default=10, help='Maximum number of clusters to test')
@click.option('--sample-size', type=int, default=None, help='Data sampling size (None uses all data)')
@click.option('--auto-k', is_flag=True, help='Automatically select optimal k using elbow method')
@click.option('--clustering-result-path', type=str, default="./outputs/step2_kmeans/clustering_result.parquet", help='Path to existing clustering result parquet file')
@click.option('--target-clusters', type=str, default="0,3", help='Specific cluster IDs to use (comma-separated, e.g. 0,1,2)')
def main(cache_dir, k_min, k_max, sample_size, auto_k, clustering_result_path, target_clusters):
    """Large-scale EMG data clustering using Spark"""
    random.seed(42)
    np.random.seed(42)
    os.environ['PYTHONHASHSEED'] = str(42)
    
    # Convert target_clusters string to list
    if target_clusters:
        target_clusters = [int(x.strip()) for x in target_clusters.split(',')]
    
    # target_clusters is required when clustering result path is provided
    if clustering_result_path and target_clusters is None:
        print("Error: target-clusters must be specified when clustering-result-path is provided.")
        return
    
    # Create Spark session
    spark = SparkSession.builder \
        .appName("Clustering") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.sql.adaptive.skewJoin.enabled", "true") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "8g") \
        .config("spark.memory.fraction", "0.8") \
        .config("spark.memory.storageFraction", "0.6") \
        .getOrCreate()
    
    try:
        df, feature_cols = load_features(spark, cache_dir, clustering_result_path, target_clusters)
        if df is None:
            print("Failed to load features")
            return
        
        # Sampling (optional)
        if sample_size and sample_size < df.count():
            df = df.sample(fraction=sample_size/df.count(), seed=42)
            print(f"Sampled to {df.count()} records")
        
        # Combine features using VectorAssembler
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        df_assembled = assembler.transform(df)
        
        k_range = list(range(k_min, k_max + 1))
        
        if auto_k:
            print("\n=== Finding optimal number of K-means clusters ===")
            elbow_k_kmeans, best_silhouette_k_kmeans, _, _ = find_optimal_clusters(
                spark, df_assembled, k_range, 'kmeans', "./outputs/kmeans"
            )

            # Perform final clustering with optimal k
            print(f"\nPerforming K-means clustering with optimal k={elbow_k_kmeans}...")
            perform_kmeans_clustering(df_assembled, elbow_k_kmeans, "./outputs/kmeans")
            
        else:
            default_k = k_min
            perform_kmeans_clustering(df_assembled, default_k, "./outputs/kmeans")
            
            
    finally:
        spark.stop()

if __name__ == '__main__':
    main() 
    