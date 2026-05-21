from pathlib import Path
import click
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeansModel
from pyspark.sql.functions import array


def load_spark_model(model_path):
    """Load a Spark clustering model"""
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")
    
    try:
        model = KMeansModel.load(str(model_path))
        print(f"Model loaded successfully: {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def load_features(spark, features_path, clustering_result_path=None, target_clusters=None):
    """Load cached features as a Spark DataFrame"""
    features_path = Path(features_path)
    if features_path.exists():
        # Read Parquet file directly as a Spark DataFrame
        df = spark.read.parquet(str(features_path))
        
        # Filter by clustering result if provided
        if clustering_result_path and target_clusters is not None:
            clustering_result_path = Path(clustering_result_path)
            if clustering_result_path.exists():
                clustering_result = spark.read.parquet(str(clustering_result_path))
                
                # Remove existing prediction column if present
                if 'prediction' in df.columns:
                    df = df.drop('prediction')
                
                # Join with clustering result and filter
                df = df.join(clustering_result.select("segmentid", "prediction"), on="segmentid", how="inner")
                df = df.filter(df.prediction.isin(target_clusters))
                df = df.drop('prediction')
                
                print(f"Filtered to clusters {target_clusters}: {df.count()} records")
            else:
                print(f"Warning: Clustering result file not found: {clustering_result_path}")
        
        # Extract feature columns (excluding segmentid)
        feature_cols = [col for col in df.columns if col.startswith('feature_')]
        feature_cols.sort()  # Sort in order: feature_0, feature_1, ...
        
        print(f"Feature file loaded successfully: {features_path}")
        print(f"Record count: {df.count():,}")
        print(f"Feature column count: {len(feature_cols)}")
        
        # Combine features using VectorAssembler
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        df_assembled = assembler.transform(df)
        
        return df_assembled, feature_cols
    else:
        print(f"Feature file not found: {features_path}")
        return None, None


def perform_inference(spark, model, df_assembled, output_path=None):
    """Perform inference using a Spark model"""
    print("Performing inference...")
    
    predictions = model.transform(df_assembled)
    
    # Convert feature_ columns into a single features list
    feature_cols = [col for col in predictions.columns if col.startswith('feature_')]
    feature_cols.sort()
    
    # Combine feature_ columns into an array to create features column
    predictions_with_features = predictions.select(
        "segmentid", 
        "prediction", 
        array(feature_cols).alias("features")
    )
    
    # Save entire data as a single parquet file (excluding features column)
    if output_path:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Delete existing file if present
        final_path = output_path / "clustering_result.parquet"
        if final_path.exists():
            if final_path.is_file():
                final_path.unlink()
            else:
                import shutil
                shutil.rmtree(final_path)
        
        # Save Spark DataFrame as a single parquet file
        predictions_with_features \
            .coalesce(1) \
            .write.mode("overwrite").parquet(str(output_path / "temp_clustering_result.parquet"))
        
        # Rename to a single parquet file
        parquet_dir = output_path / "temp_clustering_result.parquet"
        parquet_files = list(parquet_dir.glob("*.parquet"))
        if parquet_files:
            # Copy the first parquet file with the desired name
            import shutil
            shutil.copy2(parquet_files[0], final_path)
            # Delete temporary folder
            shutil.rmtree(parquet_dir)
        
        print(f"Inference results saved: {final_path}")
    
    return predictions


def analyze_cluster_distribution(predictions_df):
    """Analyze cluster distribution (using Spark DataFrame)"""
    print("\n=== Cluster distribution analysis ===")

    # Total data count
    total_count = predictions_df.count()
    print(f"Total records: {total_count:,}")

    # Per-cluster statistics (using Spark SQL)
    cluster_stats = predictions_df.groupBy("prediction").count().orderBy("prediction")

    print("\nPer-cluster detailed statistics:")
    for row in cluster_stats.collect():
        cluster_id = row['prediction']
        count = row['count']
        percentage = (count / total_count * 100)
        print(f"Cluster {cluster_id}: {count:,} ({percentage:.2f}%)")
    
    return cluster_stats


@click.command()
@click.option('--model-path', type=str, default="./outputs/step1_kmeans/kmeans_model", help='Spark model path')
@click.option('--features-path', type=str, default="./outputs/inference/step2_kmeans/features.parquet", help='Features file path')
@click.option('--clustering-result-path', type=str, default="./outputs/inference/step1_kmeans/clustering_result.parquet", help='Path to clustering result file (optional)')
@click.option('--target-clusters', type=str, default="0,5", help='Target clusters (comma separated) (optional)')
@click.option('--output-path', type=str, default="./outputs/inference/step2_kmeans", help='Output save path (optional)')
def main(model_path, features_path, clustering_result_path, target_clusters, output_path):
    """Inference using a Spark clustering model"""
    
    # Convert target_clusters string to list
    if target_clusters:
        target_clusters = [int(x.strip()) for x in target_clusters.split(',')]
    
    # target_clusters is required when clustering result path is provided
    if clustering_result_path and target_clusters is None:
        print("Error: target-clusters must be specified when clustering-result-path is provided.")
        return
    
    # Create Spark session
    spark = SparkSession.builder \
        .appName("ClusteringInference") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "8g") \
        .getOrCreate()
    
    try:
        # Load model
        print("1. Loading Spark model...")
        model = load_spark_model(model_path)
        if model is None:
            return

        # Load feature data and create vectors
        print("\n2. Loading feature data and creating vectors...")
        df_assembled, feature_cols = load_features(spark, features_path, clustering_result_path, target_clusters)
        if df_assembled is None:
            return

        # Perform inference
        print("\n3. Performing inference...")
        predictions = perform_inference(spark, model, df_assembled, output_path)

        # Analyze cluster distribution
        print("\n4. Analyzing cluster distribution...")
        analyze_cluster_distribution(predictions)

        print("\n=== Inference complete ===")
        print(f"Processed records: {predictions.count():,}")
        print(f"Feature dimension: {len(feature_cols)}")

    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        spark.stop()


if __name__ == '__main__':
    main() 
