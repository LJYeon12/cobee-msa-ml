"""
Matrix Factorization 모델 학습

Surprise 라이브러리의 SVD 알고리즘으로 추천 모델을 학습하고
MLflow에 실험 결과를 기록합니다.

실행 방법:
    python scripts/train_model.py

필요 파일:
    - data/interactions.csv (feature_engineering.py 실행 후 생성)

출력:
    - models/svd_model.pkl (학습된 모델)
    - mlruns/ (MLflow 실험 기록)
"""

import os
import sys
import pandas as pd
import pickle
from datetime import datetime
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logger import get_logger

# MLflow import
import mlflow
import mlflow.sklearn

logger = get_logger(__name__)


class MatrixFactorizationTrainer:
    """
    Matrix Factorization (SVD) 모델 학습 및 평가
    """
    
    # ========================================
    # 하이퍼파라미터 설정 및 설명
    # ========================================
    
    """
    SVD (Singular Value Decomposition) 하이퍼파라미터:
    
    1. n_factors (잠재 요인 개수) = 50
       - 의미: User와 Item을 몇 개의 잠재 특성으로 표현할 것인가
       - 예시: "아침형", "내향인", "반려동물 선호" 같은 추상적 특성
       - 값이 클수록: 복잡한 패턴 학습 가능, 과적합 위험 ↑
       - 값이 작을수록: 단순한 패턴만 학습, 과소적합 위험 ↑
       - 50 선택 이유:
         * 데이터가 적을 때(~1000개) 적절한 중간값
         * 과적합 방지하면서도 충분한 표현력 확보
         * 추천 시스템 논문에서 일반적으로 사용되는 범위 (20~100)
    
    2. n_epochs (학습 반복 횟수) = 20
       - 의미: 전체 데이터셋을 몇 번 반복해서 학습할 것인가
       - 값이 클수록: 더 정교하게 학습, 시간 ↑, 과적합 위험 ↑
       - 값이 작을수록: 빠른 학습, 과소적합 위험 ↑
       - 20 선택 이유:
         * 대부분의 추천 시스템에서 수렴하는 적절한 횟수
         * 초기 단계에서 빠른 실험을 위한 균형점
         * 나중에 early stopping 도입 전까지 안전한 기본값
    
    3. lr_all (학습률, Learning Rate) = 0.005
       - 의미: 매 학습마다 파라미터를 얼마나 크게 업데이트할 것인가
       - 값이 클수록: 빠른 학습, 불안정, 발산 위험 ↑
       - 값이 작을수록: 안정적 학습, 느림, 지역 최소값에 갇힐 위험 ↑
       - 0.005 선택 이유:
         * Surprise 라이브러리 기본값 (0.005)
         * SGD 최적화에서 검증된 안정적인 값
         * 작은 데이터셋에서도 안전하게 수렴
    
    4. reg_all (정규화 계수, Regularization) = 0.02
       - 의미: 과적합을 방지하기 위한 패널티 강도
       - 값이 클수록: 과적합 방지 ↑, 과소적합 위험 ↑
       - 값이 작을수록: 표현력 ↑, 과적합 위험 ↑
       - 0.02 선택 이유:
         * Surprise 기본값 (0.02)
         * 작은 데이터셋에서 과적합 방지 효과적
         * L2 정규화로 파라미터 값이 너무 커지는 것 방지
    
    추가 파라미터 (Surprise 기본값 사용):
    
    5. biases = True
       - User bias와 Item bias를 모델에 포함
       - 예: "이 사용자는 평균적으로 까다로움", "이 게시글은 인기 많음"
    
    6. init_mean = 0, init_std_dev = 0.1
       - 초기 파라미터 값의 평균과 표준편차
       - 정규 분포로 초기화하여 학습 안정화
    
    7. random_state = 42
       - 재현 가능성을 위한 랜덤 시드 고정
    """
    
    HYPERPARAMETERS = {
        'n_factors': 50,        # 잠재 요인 개수
        'n_epochs': 20,         # 학습 반복 횟수
        'lr_all': 0.005,        # 학습률
        'reg_all': 0.02,        # 정규화 계수
        'init_mean': 0,         # 초기값 평균
        'init_std_dev': 0.1,    # 초기값 표준편차
        'random_state': 42      # 랜덤 시드
    }
    
    def __init__(self):
        """
        초기화
        """
        
        # MLflow 설정
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("matrix-factorization")
        
        logger.info("MatrixFactorizationTrainer 초기화 완료")
    
    def load_data(self, data_path: str = "data/interactions.csv"):
        """
        Interaction matrix 로드
        
        Args:
            data_path: CSV 파일 경로
        
        Returns:
            Surprise Dataset 객체
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"데이터 파일을 찾을 수 없습니다: {data_path}\n"
                f"먼저 feature_engineering.py를 실행하세요."
            )
        
        # CSV 로드
        df = pd.read_csv(data_path)
        logger.info(f"데이터 로드 완료: {len(df)}개 상호작용")
        logger.info(f"  - 유니크 사용자: {df['user_id'].nunique()}")
        logger.info(f"  - 유니크 아이템: {df['item_id'].nunique()}")
        logger.info(f"  - Rating 범위: {df['rating'].min()} ~ {df['rating'].max()}")
        
        # Surprise Dataset 변환
        # Reader: rating 범위 지정 (1.0 ~ 5.0)
        reader = Reader(rating_scale=(1.0, 5.0))
        data = Dataset.load_from_df(df[['user_id', 'item_id', 'rating']], reader)
        
        return data
    
    def train(self, data):
        """
        SVD 모델 학습
        
        Args:
            data: Surprise Dataset
        
        Returns:
            tuple: (trained_model, trainset, testset, predictions)
        """
        logger.info("=" * 60)
        logger.info("모델 학습 시작")
        logger.info("=" * 60)
        
        # Train/Test 분할 (80/20)
        trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
        logger.info(f"학습 데이터: {trainset.n_ratings}개")
        logger.info(f"테스트 데이터: {len(testset)}개")
        
        # SVD 모델 생성
        model = SVD(**self.HYPERPARAMETERS)
        
        logger.info(f"하이퍼파라미터: {self.HYPERPARAMETERS}")
        
        # 학습 시작
        start_time = datetime.now()
        model.fit(trainset)
        training_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"학습 완료 (소요 시간: {training_time:.2f}초)")
        
        # 테스트 예측
        predictions = model.test(testset)
        
        # 평가 지표 계산
        rmse = accuracy.rmse(predictions, verbose=False)
        mae = accuracy.mae(predictions, verbose=False)
        
        logger.info(f"평가 결과:")
        logger.info(f"  - RMSE: {rmse:.4f}")
        logger.info(f"  - MAE: {mae:.4f}")
        
        return model, trainset, testset, predictions, rmse, mae, training_time
    
    def save_model(self, model, output_path: str = "models/svd_model.pkl"):
        """
        학습된 모델을 파일로 저장
        
        Args:
            model: 학습된 SVD 모델
            output_path: 저장 경로
        """
        # 디렉토리 생성
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 모델 저장
        with open(output_path, 'wb') as f:
            pickle.dump(model, f)
        
        logger.info(f"모델 저장 완료: {output_path}")
    
    def log_to_mlflow(self, model, rmse, mae, training_time):
        """
        MLflow에 실험 결과 기록
        
        Args:
            model: 학습된 모델
            rmse: RMSE 값
            mae: MAE 값
            training_time: 학습 소요 시간 (초)
        """
        with mlflow.start_run(run_name=f"SVD_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # 하이퍼파라미터 기록
            mlflow.log_params(self.HYPERPARAMETERS)
            
            # 평가 지표 기록
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("training_time_seconds", training_time)
            
            # 모델 저장
            mlflow.sklearn.log_model(model, "model")
            
            # 태그 추가
            mlflow.set_tag("model_type", "SVD")
            mlflow.set_tag("phase", "P2-P3")
            
            logger.info("MLflow 기록 완료")
    
    def run(self, data_path: str = "data/interactions.csv", output_path: str = "models/svd_model.pkl"):
        """
        전체 파이프라인 실행
        
        Args:
            data_path: 입력 데이터 경로
            output_path: 모델 저장 경로
        """
        try:
            # 1. 데이터 로드
            data = self.load_data(data_path)
            
            # 2. 모델 학습
            model, trainset, testset, predictions, rmse, mae, training_time = self.train(data)
            
            # 3. 모델 저장
            self.save_model(model, output_path)
            
            # 4. MLflow 기록
            self.log_to_mlflow(model, rmse, mae, training_time)
            
            logger.info("=" * 60)
            logger.info("학습 파이프라인 완료")
            logger.info("=" * 60)
            
            return model
            
        except Exception as e:
            logger.error(f"학습 실패: {e}", exc_info=True)
            raise


def main():
    """메인 실행 함수"""
    try:
        
        # 모델 학습
        trainer = MatrixFactorizationTrainer()
        trainer.run()
        
        logger.info("MLflow UI 확인: mlflow ui --port 5000")
        
        sys.exit(0)  # 성공
        
    except Exception as e:
        logger.error(f"학습 실패: {e}", exc_info=True)
        sys.exit(1)  # 실패


if __name__ == "__main__":
    main()