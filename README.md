# Cobee MSA ML - 룸메이트 추천 시스템

하이브리드 추천 시스템을 사용하여 최적의 룸메이트 매칭을 제공하는 머신러닝 서비스입니다.

## 프로젝트 개요

이 프로젝트는 Rule-Based 추천과 Matrix Factorization을 결합한 하이브리드 추천 시스템을 구현합니다.
사용자 상호작용 데이터에 따라 세 가지 Phase(P1, P2, P3)로 전환되며, 각 Phase마다 최적의 추천 알고리즘 조합을 사용합니다.

### 주요 기능

- **Phase 기반 하이브리드 추천**: 데이터 양에 따라 자동으로 추천 전략 조정
- **배치 데이터 수집**: 백엔드 MSA 서버에서 주기적으로 데이터 동기화
- **MLflow 통합**: 모델 실험 추적 및 버전 관리
- **Redis 캐싱**: 빠른 추천 응답을 위한 결과 캐싱
- **FastAPI 서빙**: REST API를 통한 추천 서비스 제공

## Phase 전략

| Phase | 상호작용 수 | Rule-Based | Matrix Factorization |
|-------|------------|-----------|---------------------|
| P1    | 0-99       | 100%      | 0%                  |
| P2    | 100-999    | 60%       | 40%                 |
| P3    | 1000+      | 20%       | 80%                 |

## 프로젝트 구조

```
Cobee-msa-ml/
├── config/              # 설정 파일
├── logs/                # 로그 파일
├── data/                # 데이터 저장소
├── models/              # 학습된 모델
├── scripts/             # 실행 스크립트
└── src/                 # 소스코드
    ├── data/            # 데이터 수집 및 처리
    ├── recommender/     # 추천 알고리즘
    ├── api/             # FastAPI 서버
    ├── models/          # 데이터 모델
    └── utils/           # 유틸리티
```

## 설치 방법

### 1. 저장소 클론

```bash
git clone <repository-url>
cd Cobee-msa-ml
```

### 2. 가상환경 생성 및 활성화

```bash
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate  # Windows
```

### 3. 의존성 설치

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. 환경변수 설정

```bash
cp .env.example .env
# .env 파일을 열어서 실제 값으로 수정
```

### 5. 데이터베이스 설정

PostgreSQL 데이터베이스를 준비하고 .env 파일에 접속 정보를 입력합니다.

## 실행 방법

### API 서버 실행

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

API 문서는 `http://localhost:8000/docs`에서 확인할 수 있습니다.

### 데이터 수집 (수동 실행)

```bash
python scripts/data_collector.py
```

### 모델 학습 (수동 실행)

```bash
python scripts/train_model.py
```

### Phase 업데이트 (수동 실행)

```bash
python scripts/update_phase.py
```

## 개발 가이드

### 코드 스타일

이 프로젝트는 Black 포맷터를 사용합니다.

```bash
black src/
```

### 테스트 실행

```bash
pytest
```

## 배포

배포 가이드는 추후 작성 예정입니다.

## 기술 스택

- **Python 3.10+**
- **FastAPI**: REST API 프레임워크
- **scikit-surprise**: Matrix Factorization 구현
- **MLflow**: 실험 추적 및 모델 관리
- **PostgreSQL**: 데이터 저장소
- **Redis**: 캐시 저장소
- **pandas/numpy**: 데이터 처리