# Limit OpenBLAS thread count
import os
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'

import random
from pathlib import Path
import click
import numpy as np
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans, GaussianMixture
from pyspark.ml.evaluation import ClusteringEvaluator
from visualization import sampling_for_visualization, visualize_clusters


def load_features(spark, cache_dir, step, clustering_result_path=None, target_clusters=None):
    """Load cached features as a Spark DataFrame"""
    features_parquet = Path(cache_dir) / f"step{step}_extracted_features.parquet"
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


def save_model(model, output_dir, model_name):
    """Save a trained model"""
    model_path = Path(output_dir) / f"{model_name}_model"
    model.save(str(model_path))
    print(f"Model saved to {model_path}")


def perform_kmeans_clustering(df, n_clusters, output_dir):
    """Perform K-means clustering"""
    print(f"Performing K-means clustering with {n_clusters} clusters...")
    
    # Train K-means model
    kmeans = KMeans(featuresCol="features", k=n_clusters, seed=42)
    model = kmeans.fit(df)
    predictions = model.transform(df)
    
    # Print count per cluster
    cluster_counts = predictions.groupBy("prediction").count().orderBy("prediction")
    print("\n=== K-means cluster counts ===")
    for row in cluster_counts.collect():
        print(f"Cluster {row['prediction']}: {row['count']:,}")
    print("=" * 30)
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save model
    save_model(model, output_dir, "kmeans")

    # Save cluster centers
    centers = model.clusterCenters()
    np.save(output_path / "cluster_centers.npy", np.array(centers))
    
    # Convert vectors to arrays and save
    result_pdf = predictions.select("segmentid", "prediction", "features").toPandas()
    result_pdf['features'] = result_pdf['features'].apply(lambda x: x.toArray())
    result_pdf.to_parquet(output_path / "clustering_result.parquet", index=False)

    evaluator = ClusteringEvaluator()
    silhouette = evaluator.evaluate(predictions)
    print(f"K-means Silhouette Score: {silhouette:.4f}")
    
    sampled_features, sampled_labels = sampling_for_visualization(result_pdf, max_samples=100000)
    visualize_clusters(sampled_features, sampled_labels, output_dir)

    return predictions, model


def perform_gmm_clustering(df, n_components, output_dir):
    """Perform GMM clustering"""
    print(f"Performing GMM clustering with {n_components} components...")
    
    # Train GMM model
    gmm = GaussianMixture(featuresCol="features", k=n_components, seed=42)
    model = gmm.fit(df)
    predictions = model.transform(df)
    
    # Print count per cluster
    cluster_counts = predictions.groupBy("prediction").count().orderBy("prediction")
    print("\n=== GMM cluster counts ===")
    for row in cluster_counts.collect():
        print(f"Cluster {row['prediction']}: {row['count']:,}")
    print("=" * 30)
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save model
    save_model(model, output_dir, "gmm")

    # Save GMM cluster centers (mean of Gaussian components)
    centers = model.gaussiansDF.select("mean").collect()
    cluster_centers = np.array([center['mean'].toArray() for center in centers])
    np.save(output_path / "cluster_centers.npy", cluster_centers)

    # Save GMM component information
    gmm_components = model.gaussiansDF.select("cov").collect()
    gmm_components = np.array([component['cov'].toArray() for component in gmm_components])
    np.save(output_path / "gmm_components.npy", gmm_components)
    
    # Convert vectors to arrays and save
    result_pdf = predictions.select("segmentid", "prediction", "features").toPandas()
    result_pdf['features'] = result_pdf['features'].apply(lambda x: x.toArray())
    result_pdf.to_parquet(output_path / "clustering_result.parquet", index=False)

    evaluator = ClusteringEvaluator()
    silhouette = evaluator.evaluate(predictions)
    print(f"GMM Silhouette Score: {silhouette:.4f}")
    
    sampled_features, sampled_labels = sampling_for_visualization(result_pdf, max_samples=100000)
    visualize_clusters(sampled_features, sampled_labels, output_dir)
    
    return predictions, model


@click.command()
@click.option('--cache-dir', type=str, default='./features')
@click.option('--method', type=click.Choice(['kmeans', 'gmm', 'all']), default='kmeans')
@click.option('--n-clusters', type=int, default=5)
@click.option('--step', type=int, default=2)
@click.option('--sample-size', type=int, default=None, help='Data sampling size (None uses all data)')
@click.option('--clustering-result-path', type=str, default="./outputs_v2/step1_kmeans/clustering_result.parquet", help='Path to existing clustering result parquet file')    # "./outputs/step1_kmeans/clustering_result.parquet"
@click.option('--target-clusters', type=str, default="0,4,6", help='Specific cluster IDs to use (comma-separated, e.g. 0,1,2)') # "0,5"
def main(cache_dir, method, n_clusters, step, sample_size, clustering_result_path, target_clusters):
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
        .config("spark.driver.maxResultSize", "2g") \
        .getOrCreate()
    
    try:
        df, feature_cols = load_features(spark, cache_dir, step, clustering_result_path, target_clusters)
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
        
        # Perform clustering
        if method in ['kmeans', 'all']:
            perform_kmeans_clustering(df_assembled, n_clusters, f"./outputs_v2/step{step}_kmeans")
        
        if method in ['gmm', 'all']:
            perform_gmm_clustering(df_assembled, n_clusters, f"./outputs_v2/step{step}_kmeans")
            
    finally:
        spark.stop()

if __name__ == '__main__':
    main() 
    