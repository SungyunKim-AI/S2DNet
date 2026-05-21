from pathlib import Path
import click
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeansModel
from pyspark.sql.functions import array


def load_spark_model(model_path):
    """Spark 클러스터링 모델 로드"""
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"모델 경로를 찾을 수 없습니다: {model_path}")
    
    try:
        model = KMeansModel.load(str(model_path))
        print(f"모델을 성공적으로 로드했습니다: {model_path}")
        return model
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {e}")
        return None


def load_features(spark, features_path, clustering_result_path=None, target_clusters=None):
    """캐시된 피처를 Spark DataFrame으로 로드"""
    features_path = Path(features_path)
    if features_path.exists():
        # Parquet 파일을 Spark DataFrame으로 직접 읽기
        df = spark.read.parquet(str(features_path))
        
        # 클러스터링 결과가 주어진 경우 필터링
        if clustering_result_path and target_clusters is not None:
            clustering_result_path = Path(clustering_result_path)
            if clustering_result_path.exists():
                clustering_result = spark.read.parquet(str(clustering_result_path))
                
                # 기존 prediction 컬럼이 있으면 제거
                if 'prediction' in df.columns:
                    df = df.drop('prediction')
                
                # 클러스터링 결과와 조인하여 필터링
                df = df.join(clustering_result.select("segmentid", "prediction"), on="segmentid", how="inner")
                df = df.filter(df.prediction.isin(target_clusters))
                df = df.drop('prediction')
                
                print(f"Filtered to clusters {target_clusters}: {df.count()} records")
            else:
                print(f"Warning: Clustering result file not found: {clustering_result_path}")
        
        # feature 컬럼들 추출 (segmentid 제외)
        feature_cols = [col for col in df.columns if col.startswith('feature_')]
        feature_cols.sort()  # feature_0, feature_1, ... 순서로 정렬
        
        print(f"피처 파일을 성공적으로 로드했습니다: {features_path}")
        print(f"데이터 수: {df.count():,}")
        print(f"피처 컬럼 수: {len(feature_cols)}")
        
        # 벡터 어셈블러로 피처 결합
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        df_assembled = assembler.transform(df)
        
        return df_assembled, feature_cols
    else:
        print(f"피처 파일을 찾을 수 없습니다: {features_path}")
        return None, None


def perform_inference(spark, model, df_assembled, output_path=None):
    """Spark 모델을 사용하여 인퍼런스 수행"""
    print("인퍼런스 수행 중...")
    
    predictions = model.transform(df_assembled)
    
    # feature_ 열들을 하나의 features 리스트로 변환
    feature_cols = [col for col in predictions.columns if col.startswith('feature_')]
    feature_cols.sort()
    
    # feature_ 열들을 array로 결합하여 features 컬럼 생성
    predictions_with_features = predictions.select(
        "segmentid", 
        "prediction", 
        array(feature_cols).alias("features")
    )
    
    # features 컬럼 제외하고 전체 데이터를 단일 parquet 파일로 저장
    if output_path:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 기존 파일이 있으면 삭제
        final_path = output_path / "clustering_result.parquet"
        if final_path.exists():
            if final_path.is_file():
                final_path.unlink()
            else:
                import shutil
                shutil.rmtree(final_path)
        
        # Spark DataFrame을 단일 parquet 파일로 저장
        predictions_with_features \
            .coalesce(1) \
            .write.mode("overwrite").parquet(str(output_path / "temp_clustering_result.parquet"))
        
        # 파일명을 단일 parquet 파일로 변경
        parquet_dir = output_path / "temp_clustering_result.parquet"
        parquet_files = list(parquet_dir.glob("*.parquet"))
        if parquet_files:
            # 첫 번째 parquet 파일을 원하는 이름으로 복사
            import shutil
            shutil.copy2(parquet_files[0], final_path)
            # 임시 폴더 삭제
            shutil.rmtree(parquet_dir)
        
        print(f"인퍼런스 결과가 저장되었습니다: {final_path}")
    
    return predictions


def analyze_cluster_distribution(predictions_df):
    """클러스터 분포 분석 (Spark DataFrame 사용)"""
    print("\n=== 클러스터 분포 분석 ===")
    
    # 전체 데이터 수
    total_count = predictions_df.count()
    print(f"전체 데이터 수: {total_count:,}")
    
    # 클러스터별 통계 (Spark SQL 사용)
    cluster_stats = predictions_df.groupBy("prediction").count().orderBy("prediction")
    
    print("\n클러스터별 상세 통계:")
    for row in cluster_stats.collect():
        cluster_id = row['prediction']
        count = row['count']
        percentage = (count / total_count * 100)
        print(f"클러스터 {cluster_id}: {count:,}개 ({percentage:.2f}%)")
    
    return cluster_stats


@click.command()
@click.option('--model-path', type=str, default="./outputs/step1_kmeans/kmeans_model", help='Spark 모델 경로')
@click.option('--features-path', type=str, default="./outputs/inference/step2_kmeans/features.parquet", help='features 파일 경로')
@click.option('--clustering-result-path', type=str, default="./outputs/inference/step1_kmeans/clustering_result.parquet", help='clustering result 파일 경로 (선택사항)')
@click.option('--target-clusters', type=str, default="0,5", help='target clusters (comma separated) (선택사항)')
@click.option('--output-path', type=str, default="./outputs/inference/step2_kmeans", help='결과 저장 경로 (선택사항)')
def main(model_path, features_path, clustering_result_path, target_clusters, output_path):
    """Spark 클러스터링 모델을 사용한 인퍼런스"""
    
    # target_clusters 문자열을 리스트로 변환
    if target_clusters:
        target_clusters = [int(x.strip()) for x in target_clusters.split(',')]
    
    # 클러스터링 결과 경로가 주어진 경우 target_clusters도 필요
    if clustering_result_path and target_clusters is None:
        print("Error: clustering-result-path가 주어진 경우 target-clusters도 지정해야 합니다.")
        return
    
    # Spark 세션 생성
    spark = SparkSession.builder \
        .appName("ClusteringInference") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "8g") \
        .getOrCreate()
    
    try:
        # 모델 로드
        print("1. Spark 모델 로드 중...")
        model = load_spark_model(model_path)
        if model is None:
            return
        
        # 피처 데이터 로드 및 벡터 생성
        print("\n2. 피처 데이터 로드 및 벡터 생성 중...")
        df_assembled, feature_cols = load_features(spark, features_path, clustering_result_path, target_clusters)
        if df_assembled is None:
            return
        
        # 인퍼런스 수행
        print("\n3. 인퍼런스 수행 중...")
        predictions = perform_inference(spark, model, df_assembled, output_path)
        
        # 클러스터 분포 분석
        print("\n4. 클러스터 분포 분석...")
        analyze_cluster_distribution(predictions)
        
        print("\n=== 인퍼런스 완료 ===")
        print(f"처리된 데이터 수: {predictions.count():,}")
        print(f"피처 차원: {len(feature_cols)}")
        
    except Exception as e:
        print(f"인퍼런스 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        spark.stop()


if __name__ == '__main__':
    main() 
