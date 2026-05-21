# OpenBLAS 스레드 제한 설정
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
    """캐시된 피처를 Spark DataFrame으로 로드"""
    features_parquet = Path(cache_dir) / f"step{step}_extracted_features.parquet"
    if features_parquet.exists():
        # Parquet 파일을 Spark DataFrame으로 직접 읽기
        df = spark.read.parquet(str(features_parquet))
        
        # 클러스터링 결과가 주어진 경우 필터링
        if clustering_result_path and target_clusters is not None:
            clustering_result = spark.read.parquet(str(clustering_result_path))
            if 'prediction' in df.columns:
                df = df.drop('prediction')
            df = df.join(clustering_result.select("segmentid", "prediction"), on="segmentid", how="inner")
            df = df.filter(df.prediction.isin(target_clusters))
            df = df.drop('prediction')
            
            print(f"Filtered to clusters {target_clusters}: {df.count()} records")
        
        # feature 컬럼들 추출 (segmentid 제외)
        feature_cols = [col for col in df.columns if col.startswith('feature_')]
        feature_cols.sort()  # feature_0, feature_1, ... 순서로 정렬
        
        print(f"Loaded {df.count()} features")
        print(f"Feature columns: {feature_cols}")
        return df, feature_cols
    else:
        print(f"No cached features found at {features_parquet}")
        return None, None


def save_model(model, output_dir, model_name):
    """훈련된 모델을 저장"""
    model_path = Path(output_dir) / f"{model_name}_model"
    model.save(str(model_path))
    print(f"Model saved to {model_path}")


def perform_kmeans_clustering(df, n_clusters, output_dir):
    """K-means 클러스터링 수행"""
    print(f"Performing K-means clustering with {n_clusters} clusters...")
    
    # K-means 모델 훈련
    kmeans = KMeans(featuresCol="features", k=n_clusters, seed=42)
    model = kmeans.fit(df)
    predictions = model.transform(df)
    
    # 각 클러스터의 개수 출력
    cluster_counts = predictions.groupBy("prediction").count().orderBy("prediction")
    print("\n=== K-means 클러스터별 개수 ===")
    for row in cluster_counts.collect():
        print(f"클러스터 {row['prediction']}: {row['count']:,}개")
    print("=" * 30)
    
    # 결과 저장
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 모델 저장
    save_model(model, output_dir, "kmeans")
    
    # 클러스터 중심 저장
    centers = model.clusterCenters()
    np.save(output_path / "cluster_centers.npy", np.array(centers))
    
    # 벡터를 배열로 변환하여 저장
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
    """GMM 클러스터링 수행"""
    print(f"Performing GMM clustering with {n_components} components...")
    
    # GMM 모델 훈련
    gmm = GaussianMixture(featuresCol="features", k=n_components, seed=42)
    model = gmm.fit(df)
    predictions = model.transform(df)
    
    # 각 클러스터의 개수 출력
    cluster_counts = predictions.groupBy("prediction").count().orderBy("prediction")
    print("\n=== GMM 클러스터별 개수 ===")
    for row in cluster_counts.collect():
        print(f"클러스터 {row['prediction']}: {row['count']:,}개")
    print("=" * 30)
    
    # 결과 저장
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 모델 저장
    save_model(model, output_dir, "gmm")
    
    # GMM 클러스터 중심 저장 (가우시안 컴포넌트의 평균)
    centers = model.gaussiansDF.select("mean").collect()
    cluster_centers = np.array([center['mean'].toArray() for center in centers])
    np.save(output_path / "cluster_centers.npy", cluster_centers)

    # GMM 컴포넌트 정보 저장
    gmm_components = model.gaussiansDF.select("cov").collect()
    gmm_components = np.array([component['cov'].toArray() for component in gmm_components])
    np.save(output_path / "gmm_components.npy", gmm_components)
    
    # 벡터를 배열로 변환하여 저장
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
@click.option('--sample-size', type=int, default=None, help='데이터 샘플링 크기 (None이면 전체 사용)')
@click.option('--clustering-result-path', type=str, default="./outputs_v2/step1_kmeans/clustering_result.parquet", help='기존 클러스터링 결과 parquet 파일 경로')    # "./outputs/step1_kmeans/clustering_result.parquet"
@click.option('--target-clusters', type=str, default="0,4,6", help='사용할 특정 클러스터 번호들 (쉼표로 구분, 예: 0,1,2)') # "0,5"
def main(cache_dir, method, n_clusters, step, sample_size, clustering_result_path, target_clusters):
    """Spark를 이용한 대규모 EMG 데이터 클러스터링"""
    random.seed(42)
    np.random.seed(42)
    os.environ['PYTHONHASHSEED'] = str(42)
    
    # target_clusters 문자열을 리스트로 변환
    if target_clusters:
        target_clusters = [int(x.strip()) for x in target_clusters.split(',')]

    # 클러스터링 결과 경로가 주어진 경우 target_clusters도 필요
    if clustering_result_path and target_clusters is None:
        print("Error: clustering-result-path가 주어진 경우 target-clusters도 지정해야 합니다.")
        return
    
    # Spark 세션 생성
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
        
        # 샘플링 (선택사항)
        if sample_size and sample_size < df.count():
            df = df.sample(fraction=sample_size/df.count(), seed=42)
            print(f"Sampled to {df.count()} records")
        
        # 벡터 어셈블러로 피처 결합
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        df_assembled = assembler.transform(df)
        
        # 클러스터링 수행
        if method in ['kmeans', 'all']:
            perform_kmeans_clustering(df_assembled, n_clusters, f"./outputs_v2/step{step}_kmeans")
        
        if method in ['gmm', 'all']:
            perform_gmm_clustering(df_assembled, n_clusters, f"./outputs_v2/step{step}_kmeans")
            
    finally:
        spark.stop()

if __name__ == '__main__':
    main() 
    