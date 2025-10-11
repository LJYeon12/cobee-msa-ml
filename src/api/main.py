"""
FastAPI 메인 애플리케이션
룸메이트 추천 시스템 API 서버
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time

from src.api.routers import recommendations
from src.utils.logger import get_logger
from src.utils.config_loader import config

logger = get_logger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title="Cobee 룸메이트 추천 시스템",
    description="하이브리드 추천 시스템 (Rule-Based KNN + Matrix Factorization)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 origin만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 미들웨어: 요청 로깅
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """모든 HTTP 요청을 로깅합니다."""
    start_time = time.time()
    
    logger.info(f"요청 시작: {request.method} {request.url.path}")
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    logger.info(
        f"요청 완료: {request.method} {request.url.path} "
        f"(status={response.status_code}, time={process_time:.3f}s)"
    )
    
    return response


# 예외 핸들러
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """전역 예외 처리"""
    logger.error(f"예외 발생: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "서버 내부 오류가 발생했습니다.",
            "type": type(exc).__name__
        }
    )


# 라우터 등록
app.include_router(recommendations.router)


# 루트 엔드포인트
@app.get("/")
def root():
    """API 루트 엔드포인트"""
    return {
        "service": "Cobee 룸메이트 추천 시스템",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "endpoints": {
            "recommend": "POST /api/v1/recommendations/recommend",
            "health": "GET /api/v1/recommendations/health",
            "config": "GET /api/v1/recommendations/config",
            "stats": "GET /api/v1/recommendations/stats"
        }
    }


# 시작 이벤트
@app.on_event("startup")
async def startup_event():
    """서버 시작 시 실행"""
    logger.info("=" * 50)
    logger.info("Cobee 룸메이트 추천 시스템 시작")
    logger.info("=" * 50)
    
    # Config 로드
    try:
        config.load_config()
        current_phase = config.get_current_phase()
        logger.info(f"현재 Phase: {current_phase}")
        logger.info(f"가중치: {config.get_weights()}")
    except Exception as e:
        logger.error(f"Config 로드 실패: {e}")
    
    # DB 연결 확인
    try:
        from src.utils.database import health_check
        if health_check():
            logger.info("데이터베이스 연결 성공")
        else:
            logger.warning("데이터베이스 연결 실패")
    except Exception as e:
        logger.error(f"데이터베이스 연결 확인 실패: {e}")
    
    logger.info("서버 준비 완료")


# 종료 이벤트
@app.on_event("shutdown")
async def shutdown_event():
    """서버 종료 시 실행"""
    logger.info("=" * 50)
    logger.info("Cobee 룸메이트 추천 시스템 종료")
    logger.info("=" * 50)
    
    # DB 연결 정리
    try:
        from src.utils.database import close_all_connections
        close_all_connections()
        logger.info("데이터베이스 연결 종료 완료")
    except Exception as e:
        logger.error(f"데이터베이스 연결 종료 실패: {e}")


# 개발 서버 실행 (직접 실행 시)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # 개발 모드
        log_level="info"
    )