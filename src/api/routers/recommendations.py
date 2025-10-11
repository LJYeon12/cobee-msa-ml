"""
추천 API 라우터
Phase에 따라 적절한 추천 알고리즘을 선택하여 결과를 반환합니다.
"""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from src.models.schemas import (
    RecommendationRequest,
    RecommendationResponse,
    RecommendationItem
)
from src.api.dependencies import get_database, get_config, get_knn_recommender
from src.utils.logger import get_logger
from src.utils.config_loader import ConfigLoader
from src.recommender.knn.rule_based import RuleBasedKNNRecommender

logger = get_logger(__name__)

router = APIRouter(prefix="/recommendations", tags=["Recommendations"])


@router.post("/recommend", response_model=RecommendationResponse)
def recommend_posts(
    request: RecommendationRequest,
    db: Session = Depends(get_database),
    cfg: ConfigLoader = Depends(get_config)
):
    """
    룸메이트 추천 API
    
    현재 Phase에 따라 적절한 추천 알고리즘을 사용합니다:
    - P1 (0-99): Rule-Based KNN 100%
    - P2 (100-999): Rule-Based KNN 60% + MF 40% (MF 미구현 시 KNN 100%)
    - P3 (1000+): Rule-Based KNN 20% + MF 80% (MF 미구현 시 KNN 100%)
    
    Args:
        request: 추천 요청 (user_id, limit, include_explanations)
        db: 데이터베이스 세션
        cfg: Config 인스턴스
    
    Returns:
        RecommendationResponse: 추천 결과
    """
    logger.info(f"추천 요청 수신: user_id={request.user_id}, limit={request.limit}")
    
    # 1. 현재 Phase 확인
    current_phase = cfg.get_current_phase()
    interaction_count = cfg.get_interaction_count()
    
    logger.info(f"현재 Phase: {current_phase}, 상호작용 수: {interaction_count}")
    
    # 2. Phase에 따른 추천
    try:
        if current_phase == "P1":
            # P1: Rule-Based KNN 100%
            recommendations = _recommend_with_knn_only(
                request.user_id,
                request.limit,
                request.include_explanations,
                db,
                cfg
            )
        
        elif current_phase == "P2":
            # P2: MF 모델 확인
            # TODO: MF 모델 구현 후 혼합 로직 추가
            # 현재는 KNN 100%
            recommendations = _recommend_with_knn_only(
                request.user_id,
                request.limit,
                request.include_explanations,
                db,
                cfg
            )
            logger.warning("P2 Phase이지만 MF 모델이 없어 KNN 100% 사용")
        
        elif current_phase == "P3":
            # P3: MF 모델 확인
            # TODO: MF 모델 구현 후 혼합 로직 추가
            # 현재는 KNN 100%
            recommendations = _recommend_with_knn_only(
                request.user_id,
                request.limit,
                request.include_explanations,
                db,
                cfg
            )
            logger.warning("P3 Phase이지만 MF 모델이 없어 KNN 100% 사용")
        
        else:
            raise ValueError(f"알 수 없는 Phase: {current_phase}")
        
        # 3. 응답 생성
        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=recommendations,
            total_count=len(recommendations),
            phase=current_phase,
            model_version=f"KNN-v1.0-{current_phase}"
        )
    
    except Exception as e:
        logger.error(f"추천 생성 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"추천 생성 중 오류 발생: {str(e)}")


def _recommend_with_knn_only(
    user_id: int,
    limit: int,
    include_explanations: bool,
    db: Session,
    cfg: ConfigLoader
) -> List[RecommendationItem]:
    """
    Rule-Based KNN만 사용한 추천
    
    Args:
        user_id: 사용자 ID
        limit: 추천 개수
        include_explanations: 설명 포함 여부
        db: 데이터베이스 세션
        cfg: Config 인스턴스
    
    Returns:
        List[RecommendationItem]: 추천 목록
    """
    knn_recommender = RuleBasedKNNRecommender(db, cfg)
    recommendations = knn_recommender.recommend(
        user_id=user_id,
        limit=limit,
        include_explanations=include_explanations
    )
    return recommendations


@router.get("/health")
def health_check(
    db: Session = Depends(get_database),
    cfg: ConfigLoader = Depends(get_config)
):
    """
    서버 상태 확인
    
    Returns:
        dict: 서버 상태 정보
    """
    try:
        # DB 연결 확인
        from src.utils.database import health_check as db_health
        db_status = db_health()
        
        # 현재 Phase 정보
        current_phase = cfg.get_current_phase()
        interaction_count = cfg.get_interaction_count()
        weights = cfg.get_weights()
        
        return {
            "status": "healthy" if db_status else "unhealthy",
            "database": "connected" if db_status else "disconnected",
            "phase": {
                "current": current_phase,
                "interaction_count": interaction_count,
                "weights": weights
            },
            "services": {
                "knn": "available",
                "mf": "not_implemented"
            }
        }
    
    except Exception as e:
        logger.error(f"Health check 실패: {e}")
        raise HTTPException(status_code=503, detail="서비스 이용 불가")


@router.get("/config")
def get_current_config(cfg: ConfigLoader = Depends(get_config)):
    """
    현재 설정 정보 조회
    
    Returns:
        dict: 현재 설정 정보
    """
    return {
        "phase": {
            "current": cfg.get_current_phase(),
            "interaction_count": cfg.get_interaction_count(),
            "thresholds": cfg.get("phase.thresholds")
        },
        "weights": {
            "P1": cfg.get_weights("P1"),
            "P2": cfg.get_weights("P2"),
            "P3": cfg.get_weights("P3")
        },
        "rule_based": {
            "feature_weights": cfg.get("rule_based.feature_weights"),
            "distance_metric": cfg.get("rule_based.distance_metric")
        },
        "recommendation": cfg.get("recommendation"),
        "cache": cfg.get("cache")
    }


@router.get("/stats")
def get_recommendation_stats(db: Session = Depends(get_database)):
    """
    추천 시스템 통계 정보
    
    Returns:
        dict: 통계 정보
    """
    from src.models.orm_models import (
        MemberInformationORM,
        RecruitPostORM,
        BookmarkORM,
        ApplyRecordORM
    )
    
    try:
        # 기본 통계
        total_members = db.query(MemberInformationORM).count()
        total_posts = db.query(RecruitPostORM).count()
        recruiting_posts = db.query(RecruitPostORM).filter_by(
            recruit_status="RECRUITING"
        ).count()
        
        # 상호작용 통계
        total_bookmarks = db.query(BookmarkORM).count()
        total_applies = db.query(ApplyRecordORM).count()
        total_interactions = total_bookmarks + total_applies
        
        return {
            "members": {
                "total": total_members
            },
            "posts": {
                "total": total_posts,
                "recruiting": recruiting_posts
            },
            "interactions": {
                "bookmarks": total_bookmarks,
                "applies": total_applies,
                "total": total_interactions
            }
        }
    
    except Exception as e:
        logger.error(f"통계 조회 실패: {e}")
        raise HTTPException(status_code=500, detail="통계 조회 실패")