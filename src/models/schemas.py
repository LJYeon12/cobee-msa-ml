"""
Pydantic 데이터 모델 스키마
API 요청/응답 및 데이터 검증에 사용됩니다. (springboot DTO 역할)
"""

from datetime import datetime, date
from typing import Optional, List, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field, field_validator


# ========== Enum 정의 ==========

class Gender(str, Enum):
    """성별"""
    MAN = "MAN"
    FEMALE = "FEMALE"
    NONE = "NONE"  # 선호도에서 "상관없음"


class Lifestyle(str, Enum):
    """생활 패턴"""
    MORNING = "MORNING"
    EVENING = "EVENING"


class Personality(str, Enum):
    """성격 유형"""
    INTROVERT = "INTROVERT"
    EXTROVERT = "EXTROVERT"


class RecruitStatus(str, Enum):
    """모집 상태"""
    RECRUITING = "RECRUITING"
    ON_CONTACT = "ON_CONTACT"
    RECRUIT_OVER = "RECRUIT_OVER"


class MatchStatus(str, Enum):
    """매칭 상태"""
    ON_WAIT = "ON_WAIT"
    MATCHING = "MATCHING"
    REJECTED = "REJECTED"
    MATCHED = "MATCHED"



class MemberInformation(BaseModel):
    """
    회원 정보 (기본 정보 + 선호도 + 공개 프로필)
    """
    # 기본 정보
    memberId: int
    gender: Gender
    birthDate: datetime
    
    # 선호도 (Preferred)
    preferredGender: Optional[Gender] = None
    preferredLifeStyle: Optional[Lifestyle] = None
    preferredPersonality: Optional[Personality] = None
    possibleSmoking: bool = False
    possibleSnoring: bool = False
    hasPetAllowed: bool = False
    cohabitantCount: Optional[int] = None
    preferredAgeMin: Optional[int] = Field(None, ge=19, le=34)
    preferredAgeMax: Optional[int] = Field(None, ge=19, le=34)
    
    # 본인 특성 (my)
    myLifestyle: Optional[Lifestyle] = None
    myPersonality: Optional[Personality] = None
    isSmoking: bool = False
    isSnoring: bool = False
    hasPet: bool = False
    
    class Config:
        from_attributes = True  # SQLAlchemy 모델과 호환
    
    @field_validator('preferredAgeMin', 'preferredAgeMax')
    @classmethod
    def validate_age_range(cls, v):
        """나이 범위 검증"""
        if v is not None and not (19 <= v <= 34):
            raise ValueError('나이는 19세에서 34세 사이여야 합니다.')
        return v
    
    def calculate_age(self) -> int:
        """만 나이 계산"""
        today = date.today()
        birth = self.birthDate.date() if isinstance(self.birthDate, datetime) else self.birthDate
        age = today.year - birth.year
        # 생일이 아직 안 지났으면 -1
        if (today.month, today.day) < (birth.month, birth.day):
            age -= 1
        return age


class RecruitPost(BaseModel):
    """모집 게시글"""
    recruitPostId: int
    recruitCount: int
    rentCostMin: Optional[int] = None
    rentCostMax: Optional[int] = None
    monthlyCostMin: Optional[int] = None
    monthlyCostMax: Optional[int] = None
    
    # 선호 조건
    preferredGender: Optional[Gender] = None
    preferredLifeStyle: Optional[Lifestyle] = None
    preferredPersonality: Optional[Personality] = None
    isSmoking: bool = False
    isSnoring: bool = False
    isPetAllowed: bool = False
    cohabitantCount: Optional[int] = None
    preferredAgeMin: Optional[int] = Field(None, ge=19, le=34)
    preferredAgeMax: Optional[int] = Field(None, ge=19, le=34)
    
    # 방 정보
    hasRoom: bool = False
    address: Optional[str] = None
    regionLatitude: Optional[float] = None
    regionLongitude: Optional[float] = None
    
    # 상태
    recruitStatus: RecruitStatus
    
    # 외래키
    memberId: int  # 작성자
    
    # 타임스탬프
    createdAt: datetime
    updatedAt: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class ApplyRecord(BaseModel):
    """지원 기록"""
    recordId: int
    matchStatus: MatchStatus
    submittedAt: datetime
    
    # 외래키
    memberId: int
    recruitPostId: int
    
    # 타임스탬프
    createdAt: datetime
    updatedAt: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class Bookmark(BaseModel):
    """북마크 (관심 표시)"""
    bookmarkId: int
    
    # 외래키
    memberId: int
    recruitPostId: int
    
    # 타임스탬프
    createdAt: datetime
    updatedAt: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class Comment(BaseModel):
    """댓글"""
    commentId: int
    
    # 외래키
    memberId: int
    recruitPostId: int
    
    # 타임스탬프
    createdAt: datetime
    updatedAt: Optional[datetime] = None
    
    class Config:
        from_attributes = True


# ========== API 요청/응답 스키마 ==========

class RecommendationRequest(BaseModel):
    """추천 요청"""
    user_id: int = Field(..., gt=0, description="사용자 ID")
    limit: int = Field(10, ge=1, le=100, description="추천 개수")
    include_explanations: bool = Field(True, description="추천 이유 포함 여부")


class RecommendationExplanation(BaseModel):
    """추천 이유 설명"""
    score: float = Field(..., ge=0.0, le=1.0, description="매칭 점수")
    percentage: str = Field(..., description="매칭 퍼센트 (예: 85%)")
    reasons: List[str] = Field(default_factory=list, description="일치하는 항목들")
    details: Optional[Dict[str, Any]] = Field(None, description="상세 매칭 정보")


class RecommendationItem(BaseModel):
    """추천 항목"""
    recruit_post: RecruitPost
    score: float = Field(..., ge=0.0, le=1.0, description="추천 점수")
    rank: int = Field(..., ge=1, description="추천 순위")
    explanation: Optional[RecommendationExplanation] = None


class RecommendationResponse(BaseModel):
    """추천 응답"""
    user_id: int
    recommendations: List[RecommendationItem]
    total_count: int = Field(..., description="추천된 총 개수")
    phase: str = Field(..., description="현재 Phase (P1, P2, P3)")
    model_version: Optional[str] = Field(None, description="모델 버전")
    generated_at: datetime = Field(default_factory=datetime.now, description="생성 시각")


# ========== 데이터 수집용 스키마 ==========

class BatchExportRequest(BaseModel):
    """배치 데이터 수집 요청"""
    last_sync_timestamp: Optional[datetime] = Field(
        None, 
        description="마지막 동기화 시각 (이후 변경된 데이터만 가져오기)"
    )
    include_tables: Optional[List[str]] = Field(
        None,
        description="수집할 테이블 목록 (None이면 전체)"
    )


class BatchExportResponse(BaseModel):
    """배치 데이터 수집 응답"""
    members: List[MemberInformation]
    recruit_posts: List[RecruitPost]
    apply_records: List[ApplyRecord]
    bookmarks: List[Bookmark]
    comments: List[Comment]
    export_timestamp: datetime = Field(default_factory=datetime.now)
    total_records: int


# ========== Phase 업데이트용 스키마 ==========

class PhaseUpdateRequest(BaseModel):
    """Phase 업데이트 요청"""
    interaction_count: int = Field(..., ge=0, description="현재 상호작용 총합")
    
    
class PhaseUpdateResponse(BaseModel):
    """Phase 업데이트 응답"""
    old_phase: str
    new_phase: str
    interaction_count: int
    weights: Dict[str, float]
    updated_at: datetime = Field(default_factory=datetime.now)