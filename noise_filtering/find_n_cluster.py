# OpenBLAS 스레드 제한 설정
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
    """캐시된 피처를 Spark DataFrame으로 로드"""
    features_parquet = Path(cache_dir) / "extracted_features.parquet"
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


def find_elbow_point(inertias, k_range):
    """엘보우 포인트를 찾는 함수"""
    # 이차 미분을 계산하여 급격한 변화 지점 찾기
    if len(inertias) < 3:
        return k_range[1] if len(k_range) > 1 else k_range[0]
    
    # 이차 미분 계산
    second_derivatives = []
    for i in range(1, len(inertias) - 1):
        second_deriv = inertias[i+1] - 2*inertias[i] + inertias[i-1]
        second_derivatives.append(second_deriv)
    
    # 가장 큰 이차 미분값의 인덱스 찾기
    max_second_deriv_idx = np.argmax(second_derivatives)
    elbow_k = k_range[max_second_deriv_idx + 1]  # +1 because we start from index 1
    
    return elbow_k


def plot_elbow_curve(k_range, inertias, silhouette_scores, output_dir):
    """엘보우 커브를 플롯하고 저장"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Inertia (Within-cluster sum of squares) 플롯
    ax1.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow Method for Optimal k')
    ax1.grid(True, alpha=0.3)
    
    # 엘보우 포인트 표시
    elbow_k = find_elbow_point(inertias, k_range)
    elbow_idx = k_range.index(elbow_k)
    ax1.axvline(x=elbow_k, color='red', linestyle='--', alpha=0.7, 
                label=f'Elbow point: k={elbow_k}')
    ax1.legend()
    
    # Silhouette Score 플롯
    ax2.plot(k_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Score vs Number of Clusters')
    ax2.grid(True, alpha=0.3)
    
    # 최고 실루엣 스코어 지점 표시
    best_silhouette_idx = np.argmax(silhouette_scores)
    best_silhouette_k = k_range[best_silhouette_idx]
    ax2.axvline(x=best_silhouette_k, color='red', linestyle='--', alpha=0.7,
                label=f'Best silhouette: k={best_silhouette_k}')
    ax2.legend()
    
    plt.tight_layout()
    
    # 저장
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path / "elbow_curve.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return elbow_k, best_silhouette_k


def find_optimal_clusters(spark, df, k_range, method='kmeans', output_dir="./outputs"):
    """최적의 클러스터 개수를 찾는 함수"""
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
            # GMM의 경우 inertia 대신 negative log likelihood 사용
            inertia = -model.summary.logLikelihood
        
        # Silhouette score 계산
        evaluator = ClusteringEvaluator()
        silhouette = evaluator.evaluate(predictions)
        
        inertias.append(inertia)
        silhouette_scores.append(silhouette)
        
        print(f"  k={k}: Inertia={inertia:.4f}, Silhouette={silhouette:.4f}")
    
    # 엘보우 커브 플롯 및 최적 k 찾기
    elbow_k, best_silhouette_k = plot_elbow_curve(k_range, inertias, silhouette_scores, output_dir)
    
    print(f"\nResults:")
    print(f"Elbow method suggests k={elbow_k}")
    print(f"Best silhouette score at k={best_silhouette_k}")
    
    # 결과를 CSV로 저장
    results_df = spark.createDataFrame([
        (k, inertia, silhouette) for k, inertia, silhouette in zip(k_range, inertias, silhouette_scores)
    ], ["k", "inertia", "silhouette_score"])
    
    output_path = Path(output_dir)
    results_df.toPandas().to_csv(output_path / f"{method}_clustering_analysis.csv", index=False)
    
    return elbow_k, best_silhouette_k, inertias, silhouette_scores



@click.command()
@click.option('--cache-dir', type=str, default='./features')
@click.option('--k-min', type=int, default=2, help='테스트할 최소 클러스터 개수')
@click.option('--k-max', type=int, default=10, help='테스트할 최대 클러스터 개수')
@click.option('--sample-size', type=int, default=None, help='데이터 샘플링 크기 (None이면 전체 사용)')
@click.option('--auto-k', is_flag=True, help='엘보우 메서드로 최적 k 자동 선택')
@click.option('--clustering-result-path', type=str, default="./outputs/step2_kmeans/clustering_result.parquet", help='기존 클러스터링 결과 parquet 파일 경로')
@click.option('--target-clusters', type=str, default="0,3", help='사용할 특정 클러스터 번호들 (쉼표로 구분, 예: 0,1,2)')
def main(cache_dir, k_min, k_max, sample_size, auto_k, clustering_result_path, target_clusters):
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
        .getOrCreate()
    
    try:
        df, feature_cols = load_features(spark, cache_dir, clustering_result_path, target_clusters)
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
        
        k_range = list(range(k_min, k_max + 1))
        
        if auto_k:
            print("\n=== K-means 최적 클러스터 개수 찾기 ===")
            elbow_k_kmeans, best_silhouette_k_kmeans, _, _ = find_optimal_clusters(
                spark, df_assembled, k_range, 'kmeans', "./outputs/kmeans"
            )
            
            # 최적 k로 최종 클러스터링 수행
            print(f"\n최적 k={elbow_k_kmeans}로 K-means 클러스터링 수행...")
            perform_kmeans_clustering(df_assembled, elbow_k_kmeans, "./outputs/kmeans")
            
        else:
            default_k = k_min
            perform_kmeans_clustering(df_assembled, default_k, "./outputs/kmeans")
            
            
    finally:
        spark.stop()

if __name__ == '__main__':
    main() 
    