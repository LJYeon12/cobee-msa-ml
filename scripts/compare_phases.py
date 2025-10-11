"""
Phase 비교 평가기

P1, P2, P3의 성능을 비교하여 최적의 Phase를 선택합니다.

실행 방법:
    python scripts/compare_phases.py

출력:
    - 각 Phase의 성능 지표
    - 권장 Phase
    - config.json 업데이트
"""

import os
import sys
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from sqlalchemy.orm import Session

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logger import get_logger
from src.utils.config_loader import ConfigLoader
from src.utils.database import SessionLocal
from src.recommender.hybrid_recommender import HybridRecommender

logger = get_logger(__name__)


class PhaseComparator:
    """
    Phase 간 성능 비교 및 최적 Phase 선택
    """
    
    def __init__(self):
        """
        초기화
        """
        self.config = ConfigLoader()
        self.config.load_config()
        
        # DB 세션
        self.db = SessionLocal()
        
        logger.info("PhaseComparator 초기화 완료")
    
    def load_test_data(self, data_path: str = "data/interactions.csv"):
        """
        테스트 데이터 로드
        
        Args:
            data_path: CSV 파일 경로
        
        Returns:
            tuple: (Surprise Dataset, testset, user_relevant_items)
        """
        df = pd.read_csv(data_path)
        
        reader = Reader(rating_scale=(1.0, 5.0))
        data = Dataset.load_from_df(df[['user_id', 'item_id', 'rating']], reader)
        
        # Train/Test 분할 (평가용)
        trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
        
        # 테스트셋을 user별로 그룹화 (relevant items)
        user_relevant_items = defaultdict(set)
        for uid, iid, rating in testset:
            if rating >= 4.0:  # rating 4.0 이상이면 relevant
                user_relevant_items[uid].add(iid)
        
        logger.info(f"테스트 데이터 로드 완료: {len(testset)}개")
        logger.info(f"평가 대상 사용자: {len(user_relevant_items)}명")
        
        return data, testset, user_relevant_items
    
    def precision_at_k(self, recommended: list, relevant: set, k: int) -> float:
        """Precision@K 계산"""
        recommended_k = recommended[:k]
        num_relevant_and_recommended = len(set(recommended_k) & relevant)
        return num_relevant_and_recommended / k if k > 0 else 0.0
    
    def recall_at_k(self, recommended: list, relevant: set, k: int) -> float:
        """Recall@K 계산"""
        recommended_k = recommended[:k]
        num_relevant_and_recommended = len(set(recommended_k) & relevant)
        num_relevant = len(relevant)
        return num_relevant_and_recommended / num_relevant if num_relevant > 0 else 0.0
    
    def ndcg_at_k(self, recommended: list, relevant: set, k: int) -> float:
        """NDCG@K 계산"""
        recommended_k = recommended[:k]
        
        # DCG 계산
        dcg = 0.0
        for i, item_id in enumerate(recommended_k, start=1):
            if item_id in relevant:
                dcg += 1.0 / np.log2(i + 1)
        
        # IDCG 계산
        idcg = 0.0
        for i in range(1, min(len(relevant), k) + 1):
            idcg += 1.0 / np.log2(i + 1)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def evaluate_phase(
        self,
        phase: str,
        user_relevant_items: dict,
        k: int = 10
    ) -> dict:
        """
        특정 Phase로 평가
        
        Args:
            phase: "P1", "P2", "P3"
            user_relevant_items: {user_id: set(relevant_item_ids)}
            k: Top-K
        
        Returns:
            dict: 평가 결과
        """
        logger.info(f"=" * 60)
        logger.info(f"{phase} Phase 평가 시작")
        logger.info(f"=" * 60)
        
        # Config를 임시로 해당 Phase로 설정
        original_phase = self.config.get_current_phase()
        self.config._config['phase']['current'] = phase
        
        # 하이브리드 추천기 생성
        recommender = HybridRecommender(self.db, self.config)
        
        precision_list = []
        recall_list = []
        ndcg_list = []
        
        for user_id, relevant_items in user_relevant_items.items():
            try:
                # 추천 생성
                recommendations = recommender.recommend(
                    user_id=user_id,
                    limit=k,
                    include_explanations=False
                )
                
                recommended_ids = [
                    item.recruit_post.recruitPostId 
                    for item in recommendations
                ]
                
                # 지표 계산
                precision = self.precision_at_k(recommended_ids, relevant_items, k)
                recall = self.recall_at_k(recommended_ids, relevant_items, k)
                ndcg = self.ndcg_at_k(recommended_ids, relevant_items, k)
                
                precision_list.append(precision)
                recall_list.append(recall)
                ndcg_list.append(ndcg)
                
            except Exception as e:
                logger.error(f"사용자 {user_id} 평가 실패: {e}")
                continue
        
        # Config 복원
        self.config._config['phase']['current'] = original_phase
        
        # 평균 계산
        results = {
            'phase': phase,
            'precision_at_10': np.mean(precision_list) if precision_list else 0.0,
            'recall_at_10': np.mean(recall_list) if recall_list else 0.0,
            'ndcg_at_10': np.mean(ndcg_list) if ndcg_list else 0.0,
            'evaluated_users': len(precision_list)
        }
        
        logger.info(f"{phase} 평가 완료:")
        logger.info(f"  - Precision@10: {results['precision_at_10']:.4f}")
        logger.info(f"  - Recall@10: {results['recall_at_10']:.4f}")
        logger.info(f"  - NDCG@10: {results['ndcg_at_10']:.4f}")
        
        return results
    
    def calculate_weighted_score(self, metrics: dict) -> float:
        """
        가중 평균 점수 계산
        
        Args:
            metrics: {precision_at_10, recall_at_10, ndcg_at_10}
        
        Returns:
            float: 종합 점수
        """
        weights = self.config.get("phase.transition_criteria.metrics", {})
        
        precision_weight = weights.get("precision_at_10", {}).get("weight", 0.5)
        recall_weight = weights.get("recall_at_10", {}).get("weight", 0.3)
        ndcg_weight = weights.get("ndcg_at_10", {}).get("weight", 0.2)
        
        score = (
            metrics['precision_at_10'] * precision_weight +
            metrics['recall_at_10'] * recall_weight +
            metrics['ndcg_at_10'] * ndcg_weight
        )
        
        return score
    
    def compare_and_recommend(
        self,
        results: dict,
        current_phase: str
    ) -> str:
        """
        Phase 비교 후 권장 Phase 반환
        
        Args:
            results: {phase: metrics}
            current_phase: 현재 Phase
        
        Returns:
            str: 권장 Phase
        """
        logger.info("=" * 60)
        logger.info("Phase 비교 분석")
        logger.info("=" * 60)
        
        # 각 Phase의 종합 점수 계산
        scores = {}
        for phase, metrics in results.items():
            score = self.calculate_weighted_score(metrics)
            scores[phase] = score
            logger.info(f"{phase}: 종합 점수 = {score:.4f}")
        
        # 최소 개선율
        min_improvement = self.config.get(
            "phase.transition_criteria.min_improvement_ratio", 
            1.1
        )
        
        # 현재 Phase 점수
        current_score = scores.get(current_phase, 0.0)
        
        # 다음 Phase 후보
        phase_order = ["P1", "P2", "P3"]
        current_idx = phase_order.index(current_phase)
        
        # 상향 평가 (P1 → P2 또는 P2 → P3)
        if current_idx < len(phase_order) - 1:
            next_phase = phase_order[current_idx + 1]
            next_score = scores.get(next_phase, 0.0)
            
            improvement_ratio = next_score / current_score if current_score > 0 else 0
            
            logger.info(f"\n개선율 분석:")
            logger.info(f"  {current_phase} 점수: {current_score:.4f}")
            logger.info(f"  {next_phase} 점수: {next_score:.4f}")
            logger.info(f"  개선율: {improvement_ratio:.2f}x")
            logger.info(f"  필요 개선율: {min_improvement:.2f}x")
            
            if improvement_ratio >= min_improvement:
                logger.info(f"\n✅ {next_phase}로 전환 권장!")
                return next_phase
            else:
                logger.info(f"\n⚠️ {current_phase} 유지 권장 (개선 불충분)")
                return current_phase
        
        # 이미 최상위 Phase
        logger.info(f"\n✅ 현재 {current_phase} 유지 (최상위 Phase)")
        return current_phase
    
    def update_config_with_results(self, results: dict, recommended_phase: str):
        """
        config.json에 평가 결과 기록
        
        Args:
            results: {phase: metrics}
            recommended_phase: 권장 Phase
        """
        now = datetime.now().isoformat()
        interaction_count = self.config.get("phase.interaction_count", 0)
        
        for phase, metrics in results.items():
            self.config._config['phase']['evaluation_history'][phase] = {
                'last_evaluated': now,
                'precision_at_10': metrics['precision_at_10'],
                'recall_at_10': metrics['recall_at_10'],
                'ndcg_at_10': metrics['ndcg_at_10'],
                'interaction_count_at_eval': interaction_count
            }
        
        # config 저장
        import json
        config_path = self.config._config_path
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config._config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"config.json 업데이트 완료")
    
    def run(self):
        """
        전체 비교 평가 실행
        """
        try:
            logger.info("=" * 60)
            logger.info("Phase 비교 평가 시작")
            logger.info("=" * 60)
            
            # 1. 테스트 데이터 로드
            data, testset, user_relevant_items = self.load_test_data()
            
            if len(user_relevant_items) < 5:
                logger.warning("평가 대상 사용자가 너무 적습니다 (최소 5명 필요)")
                return
            
            # 2. 현재 Phase 확인
            current_phase = self.config.get_current_phase()
            interaction_count = self.config.get("phase.interaction_count", 0)
            
            logger.info(f"현재 Phase: {current_phase}")
            logger.info(f"상호작용 수: {interaction_count}")
            
            # 3. 평가할 Phase 결정
            phases_to_evaluate = [current_phase]
            
            # 다음 Phase도 평가 (상향 검토)
            phase_order = ["P1", "P2", "P3"]
            current_idx = phase_order.index(current_phase)
            if current_idx < len(phase_order) - 1:
                phases_to_evaluate.append(phase_order[current_idx + 1])
            
            # 4. 각 Phase 평가
            results = {}
            for phase in phases_to_evaluate:
                results[phase] = self.evaluate_phase(phase, user_relevant_items, k=10)
            
            # 5. 비교 및 권장
            recommended_phase = self.compare_and_recommend(results, current_phase)
            
            # 6. config 업데이트
            self.update_config_with_results(results, recommended_phase)
            
            logger.info("=" * 60)
            logger.info(f"최종 권장 Phase: {recommended_phase}")
            logger.info("=" * 60)
            
            return recommended_phase
            
        except Exception as e:
            logger.error(f"Phase 비교 평가 실패: {e}", exc_info=True)
            raise
        finally:
            self.db.close()


def main():
    """메인 실행 함수"""
    try:
        comparator = PhaseComparator()
        recommended_phase = comparator.run()
        
        print(f"\n🎯 권장 Phase: {recommended_phase}")
        
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"실행 실패: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()