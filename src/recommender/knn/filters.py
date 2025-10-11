"""
KNN 추천을 위한 필터링 로직
하드 제약 조건을 체크합니다.
"""

from src.models.orm_models import MemberInformationORM


def is_gender_compatible(
    member: MemberInformationORM,
    post_author: MemberInformationORM,
    post_preferred_gender: str
) -> bool:
    """
    성별 호환성 체크 (양방향)
    
    조건 1: 회원의 성별이 게시글의 선호 성별과 맞는가?
    조건 2: 게시글 작성자의 성별이 회원의 선호 성별과 맞는가?
    
    Args:
        member: 추천을 받는 회원
        post_author: 게시글 작성자
        post_preferred_gender: 게시글의 선호 성별
    
    Returns:
        bool: 양방향 모두 호환되면 True
    """
    # 조건 1: 회원 성별 vs 게시글 선호 성별
    if post_preferred_gender != "NONE" and member.gender != post_preferred_gender:
        return False
    
    # 조건 2: 게시글 작성자 성별 vs 회원 선호 성별
    if member.preferred_gender and member.preferred_gender != "NONE":
        if post_author.gender != member.preferred_gender:
            return False
    
    return True


def is_age_compatible(
    member_age: int,
    preferred_age_min: int,
    preferred_age_max: int
) -> bool:
    """
    나이 호환성 체크
    
    Args:
        member_age: 회원의 만 나이
        preferred_age_min: 선호 최소 나이
        preferred_age_max: 선호 최대 나이
    
    Returns:
        bool: 나이가 범위 내에 있으면 True
    """
    if preferred_age_min is None or preferred_age_max is None:
        return True  # 나이 선호가 없으면 통과
    
    return preferred_age_min <= member_age <= preferred_age_max


def is_age_compatible_bidirectional(
    member: MemberInformationORM,
    post_author: MemberInformationORM,
    post_preferred_age_min: int,
    post_preferred_age_max: int
) -> bool:
    """
    나이 호환성 양방향 체크
    
    조건 1: 회원의 나이가 게시글의 선호 나이 범위 내인가?
    조건 2: 작성자의 나이가 회원의 선호 나이 범위 내인가?
    
    Args:
        member: 추천을 받는 회원
        post_author: 게시글 작성자
        post_preferred_age_min: 게시글의 선호 최소 나이
        post_preferred_age_max: 게시글의 선호 최대 나이
    
    Returns:
        bool: 양방향 모두 호환되면 True
    """
    member_age = member.calculate_age()
    author_age = post_author.calculate_age()
    
    # 조건 1: 회원 나이 vs 게시글 선호 나이
    if not is_age_compatible(member_age, post_preferred_age_min, post_preferred_age_max):
        return False
    
    # 조건 2: 작성자 나이 vs 회원 선호 나이
    if member.preferred_age_min and member.preferred_age_max:
        if not is_age_compatible(author_age, member.preferred_age_min, member.preferred_age_max):
            return False
    
    return True