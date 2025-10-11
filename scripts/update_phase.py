"""
Phase 자동 업데이트

상호작용 수를 집계하고, 조건에 따라 Phase를 자동 전환합니다.

실행 방법:
    python scripts/update_phase.py

cron 설정 (매일 새벽 4:50):
    50 4 * * * cd /path/to/cobee-msa-ml && python scripts/update_phase.py
"""

import os
import sys
import json
from datetime import datetime
from sqlalchemy import text

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logger import get_logger
from src.utils.config_loader import ConfigLoader
from src.utils.database import SessionLocal

logger = get_logger(__name__)


class PhaseUpdater:
    """
    Phase 자동 업데이트 시스템
    """
    
    def __init__(self):
        """
        초기화
        """
        self.config = ConfigLoader()
        self.config.load_config()
        
        self.db = SessionLocal()
        
        logger.info("PhaseUpdater 초기화 완료")
    
    def count_interactions(self) -> int:
        """
        현재 상호작용 총합 계산
        
        상호작용 = ApplyRecord + Bookmark
        
        Returns:
            int: 상호작용 총합
        """
        try:
            # ApplyRecord 수
            apply_count = self.db.execute(
                text("SELECT COUNT(*) FROM apply_record")
            ).scalar()
            
            # Bookmark 수
            bookmark_count = self.db.execute(
                text("SELECT COUNT(*) FROM bookmark")
            ).scalar()
            
            total = apply_count + bookmark_count
            
            logger.info(f"상호작용 집계: ApplyRecord={apply_count}, Bookmark={bookmark_count}, 총합={total}")
            
            return total
            
        except Exception as e:
            logger.error(f"상호작용 집계 실패: {e}")
            return 0
    
    def should_evaluate_phase_transition(self, current_count: int, last_count: int) -> bool:
        """
        Phase 전환 평가 필요 여부 확인
        
        Args:
            current_count: 현재 상호작용 수
            last_count: 마지막 평가 시점 상호작용 수
        
        Returns:
            bool: 평가 필요 여부
        """
        # 자동 전환 비활성화 시
        if not self.config.get("phase.auto_transition_enabled", True):
            logger.info("자동 Phase 전환이 비활성화되어 있습니다")
            return False
        
        # 최소 상호작용 수
        min_interactions = self.config.get(
            "phase.thresholds.P2.min", 
            100
        )
        
        if current_count < min_interactions:
            logger.info(f"상호작용 수 부족: {current_count} < {min_interactions}")
            return False
        
        # 평가 주기 (100개 단위)
        evaluation_interval = self.config.get(
            "phase.transition_criteria.evaluation_interval",
            100
        )
        
        # 100개 단위로 증가했는지 확인
        current_milestone = (current_count // evaluation_interval) * evaluation_interval
        last_milestone = (last_count // evaluation_interval) * evaluation_interval
        
        if current_milestone > last_milestone:
            logger.info(f"평가 주기 도달: {last_milestone} → {current_milestone}")
            return True
        
        return False
    
    def get_phase_by_interaction_count(self, count: int) -> str:
        """
        상호작용 수 기반 Phase 결정 (기본 규칙)
        
        Args:
            count: 상호작용 수
        
        Returns:
            str: Phase ("P1", "P2", "P3")
        """
        thresholds = self.config.get("phase.thresholds", {})
        
        p2_min = thresholds.get("P2", {}).get("min", 100)
        p3_min = thresholds.get("P3", {}).get("min", 1000)
        
        if count >= p3_min:
            return "P3"
        elif count >= p2_min:
            return "P2"
        else:
            return "P1"
    
    def update_phase_in_config(self, new_phase: str, interaction_count: int):
        """
        config.json의 Phase 업데이트
        
        Args:
            new_phase: 새로운 Phase
            interaction_count: 현재 상호작용 수
        """
        old_phase = self.config.get_current_phase()
        
        self.config._config['phase']['current'] = new_phase
        self.config._config['phase']['interaction_count'] = interaction_count
        self.config._config['last_updated'] = datetime.now().isoformat()
        
        # 파일에 저장
        config_path = self.config._config_path
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config._config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Phase 업데이트 완료: {old_phase} → {new_phase} (상호작용: {interaction_count})")
    
    def run_phase_comparison(self) -> str:
        """
        Phase 비교 평가 실행
        
        Returns:
            str: 권장 Phase
        """
        logger.info("Phase 비교 평가 시작...")
        
        try:
            # compare_phases.py 실행
            from scripts.compare_phases import PhaseComparator
            
            comparator = PhaseComparator()
            recommended_phase = comparator.run()
            
            logger.info(f"Phase 비교 완료: 권장 Phase = {recommended_phase}")
            
            return recommended_phase
            
        except Exception as e:
            logger.error(f"Phase 비교 실패: {e}", exc_info=True)
            # 실패 시 현재 Phase 유지
            return self.config.get_current_phase()
    
    def run(self):
        """
        Phase 업데이트 메인 로직
        """
        try:
            logger.info("=" * 60)
            logger.info("Phase 업데이트 시작")
            logger.info("=" * 60)
            
            # 1. 상호작용 수 집계
            current_interaction_count = self.count_interactions()
            last_interaction_count = self.config.get("phase.interaction_count", 0)
            current_phase = self.config.get_current_phase()
            
            logger.info(f"현재 Phase: {current_phase}")
            logger.info(f"상호작용 수: {last_interaction_count} → {current_interaction_count}")
            
            # 2. Phase 전환 평가 필요 여부 확인
            should_evaluate = self.should_evaluate_phase_transition(
                current_interaction_count,
                last_interaction_count
            )
            
            new_phase = current_phase
            
            if should_evaluate:
                logger.info("🔍 성능 기반 Phase 비교 평가 실행...")
                
                # 3. 성능 비교 평가
                recommended_phase = self.run_phase_comparison()
                
                if recommended_phase != current_phase:
                    logger.info(f"✅ Phase 전환 권장: {current_phase} → {recommended_phase}")
                    new_phase = recommended_phase
                else:
                    logger.info(f"⚠️ 현재 Phase 유지 권장: {current_phase}")
                    new_phase = current_phase
            
            else:
                # 자동 전환 비활성화 시 기본 규칙 사용
                if not self.config.get("phase.auto_transition_enabled", True):
                    new_phase = self.get_phase_by_interaction_count(current_interaction_count)
                    
                    if new_phase != current_phase:
                        logger.info(f"📊 상호작용 수 기반 Phase 전환: {current_phase} → {new_phase}")
            
            # 4. config.json 업데이트
            if new_phase != current_phase or current_interaction_count != last_interaction_count:
                self.update_phase_in_config(new_phase, current_interaction_count)
            else:
                # 상호작용 수만 업데이트
                self.config._config['phase']['interaction_count'] = current_interaction_count
                config_path = self.config._config_path
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(self.config._config, f, indent=2, ensure_ascii=False)
                logger.info("상호작용 수만 업데이트 완료")
            
            logger.info("=" * 60)
            logger.info(f"Phase 업데이트 완료: {new_phase}")
            logger.info(f"상호작용 총합: {current_interaction_count}")
            logger.info("=" * 60)
            
            return new_phase
            
        except Exception as e:
            logger.error(f"Phase 업데이트 실패: {e}", exc_info=True)
            raise
        finally:
            self.db.close()


def main():
    """메인 실행 함수"""
    try:
        updater = PhaseUpdater()
        new_phase = updater.run()
        
        print(f"\n✅ Phase: {new_phase}")
        
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"실행 실패: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()