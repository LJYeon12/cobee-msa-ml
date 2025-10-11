"""
Feature Engineering: User-Item Interaction Matrix 생성

PostgreSQL에서 상호작용 데이터를 읽어와서
Matrix Factorization 학습용 interaction matrix를 생성합니다.

실행 방법:
    python scripts/feature_engineering.py

출력:
    - data/interactions.csv: user_id, item_id, rating 형식
"""

import os
import sys
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logger import get_logger
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

from src.utils.logger import get_logger

logger = get_logger(__name__)


class InteractionMatrixBuilder:
    """
    상호작용 데이터를 Matrix Factorization용 형식으로 변환
    """
    
    # 평점 매핑 규칙
    RATING_MAP = {
        # ApplyRecord match_status → rating
        'MATCHED': 5.0,    # 최종 매칭 성공 (강한 긍정 신호)
        'MATCHING': 4.0,   # 승인됨 (긍정 신호)
        'ON_WAIT': 3.0,    # 대기 중 (중립적 관심)
        'REJECTED': 1.0,   # 거절됨 (부정 신호)
        
        # Bookmark → rating
        'BOOKMARK': 4.0,   # 관심 표시 (긍정 신호)
    }
    
    def __init__(self):
        """
        환경변수에서 DB 연결 정보 로드
        """
        # DB 연결 정보
        db_host = os.getenv("POSTGRES_HOST")
        db_port = os.getenv("POSTGRES_PORT")
        db_name = os.getenv("POSTGRES_DB")
        db_user = os.getenv("POSTGRES_USER")
        db_password = os.getenv("POSTGRES_PASSWORD")
        
        # SQLAlchemy 엔진 생성
        connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        self.engine = create_engine(connection_string)
        
        logger.info(f"DB 연결: {db_host}:{db_port}/{db_name}")
    
    def load_apply_records(self) -> pd.DataFrame:
        """
        ApplyRecord 테이블에서 상호작용 데이터 로드
        
        Returns:
            DataFrame: [user_id, item_id, rating, timestamp]
        """
        query = """
        SELECT 
            member_id as user_id,
            recruit_post_id as item_id,
            match_status,
            submitted_at as timestamp
        FROM apply_record
        ORDER BY submitted_at
        """
        
        df = pd.read_sql(query, self.engine)
        
        # match_status를 rating으로 변환
        df['rating'] = df['match_status'].map(self.RATING_MAP)
        
        # 매핑되지 않은 status가 있으면 경고
        unmapped = df[df['rating'].isna()]
        if not unmapped.empty:
            logger.warning(f"매핑되지 않은 match_status 발견: {unmapped['match_status'].unique()}")
            df = df.dropna(subset=['rating'])
        
        df = df[['user_id', 'item_id', 'rating', 'timestamp']]
        logger.info(f"ApplyRecord 로드 완료: {len(df)}개")
        
        return df
    
    def load_bookmarks(self) -> pd.DataFrame:
        """
        Bookmark 테이블에서 상호작용 데이터 로드
        
        Returns:
            DataFrame: [user_id, item_id, rating, timestamp]
        """
        query = """
        SELECT 
            member_id as user_id,
            recruit_post_id as item_id,
            created_at as timestamp
        FROM bookmark
        ORDER BY created_at
        """
        
        df = pd.read_sql(query, self.engine)
        
        # 북마크는 고정 rating
        df['rating'] = self.RATING_MAP['BOOKMARK']
        
        df = df[['user_id', 'item_id', 'rating', 'timestamp']]
        logger.info(f"Bookmark 로드 완료: {len(df)}개")
        
        return df
    
    def merge_interactions(self, apply_df: pd.DataFrame, bookmark_df: pd.DataFrame) -> pd.DataFrame:
        """
        ApplyRecord와 Bookmark를 병합하고 중복 처리
        
        중복 규칙:
        - 같은 (user_id, item_id)가 apply와 bookmark 둘 다 있으면
        - apply의 rating을 우선 (더 명시적인 신호)
        
        Args:
            apply_df: ApplyRecord 데이터
            bookmark_df: Bookmark 데이터
        
        Returns:
            DataFrame: 병합된 interaction matrix
        """
        # 1. 두 데이터프레임 합치기
        all_interactions = pd.concat([apply_df, bookmark_df], ignore_index=True)
        
        # 2. (user_id, item_id) 기준으로 정렬 (timestamp 오름차순)
        all_interactions = all_interactions.sort_values(['user_id', 'item_id', 'timestamp'])
        
        # 3. 중복 제거: 같은 (user_id, item_id)는 가장 최근 것만 유지
        # apply가 bookmark보다 나중이면 apply가 남고, 아니면 첫 번째 것 유지
        all_interactions = all_interactions.drop_duplicates(
            subset=['user_id', 'item_id'], 
            keep='last'  # 가장 최근 기록 유지
        )
        
        logger.info(f"병합 완료: {len(all_interactions)}개 (중복 제거 후)")
        
        return all_interactions
    
    def save_interactions(self, df: pd.DataFrame, output_path: str):
        """
        Interaction matrix를 CSV로 저장
        
        Args:
            df: interaction 데이터프레임
            output_path: 저장 경로
        """
        # 디렉토리 생성
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # CSV 저장 (Surprise 라이브러리 형식: user_id, item_id, rating)
        df[['user_id', 'item_id', 'rating']].to_csv(output_path, index=False)
        
        logger.info(f"Interaction matrix 저장 완료: {output_path}")
        logger.info(f"  - 총 상호작용 수: {len(df)}")
        logger.info(f"  - 유니크 사용자 수: {df['user_id'].nunique()}")
        logger.info(f"  - 유니크 아이템 수: {df['item_id'].nunique()}")
        logger.info(f"  - 평균 rating: {df['rating'].mean():.2f}")
        logger.info(f"  - Rating 분포:\n{df['rating'].value_counts().sort_index()}")
    
    def build(self, output_path: str = "data/interactions.csv"):
        """
        전체 파이프라인 실행
        
        Args:
            output_path: 출력 파일 경로
        """
        logger.info("=" * 60)
        logger.info("Feature Engineering 시작")
        logger.info("=" * 60)
        
        # 1. 데이터 로드
        apply_df = self.load_apply_records()
        bookmark_df = self.load_bookmarks()
        
        # 2. 병합 및 중복 제거
        interactions_df = self.merge_interactions(apply_df, bookmark_df)
        
        # 3. 저장
        self.save_interactions(interactions_df, output_path)
        
        logger.info("=" * 60)
        logger.info("Feature Engineering 완료")
        logger.info("=" * 60)
        
        return interactions_df

def main():
    """메인 실행 함수"""
    try:
        # Interaction Matrix 생성
        builder = InteractionMatrixBuilder()
        builder.build(output_path="data/interactions.csv")
        
        sys.exit(0)  # 성공
        
    except Exception as e:
        logger.error(f"Feature Engineering 실패: {e}", exc_info=True)
        sys.exit(1)  # 실패


if __name__ == "__main__":
    main()