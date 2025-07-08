# DVC (Data Version Control) 실습 프로젝트

이 프로젝트는 **DVC (Data Version Control)**를 사용한 머신러닝 워크플로우의 실습을 위한 예제입니다. DVC의 핵심 기능들을 체험하고 학습할 수 있도록 설계되었습니다.

## 🎯 프로젝트 목표

- **DVC의 핵심 개념 이해**: 데이터 버전 관리, 파이프라인 관리, 실험 추적
- **실제 머신러닝 워크플로우 경험**: 데이터 처리 → 모델 훈련 → 평가 → 실험 관리
- **버전 관리 시스템 활용**: Git과 DVC의 연동을 통한 완전한 재현 가능한 실험

## 📚 DVC 이론적 원리

### 1. 데이터 버전 관리 (Data Versioning)

DVC는 Git이 코드를 관리하는 것처럼 데이터를 관리합니다:

- **해시 기반 추적**: 데이터 파일의 내용을 기반으로 고유한 해시값 생성
- **메타데이터 저장**: 실제 데이터는 별도 저장소에, 메타데이터만 Git에 저장
- **변경 감지**: 데이터 내용이 변경되면 자동으로 새로운 버전으로 인식

```bash
# 데이터 파일을 DVC로 추적
dvc add data/raw_data.csv

# .dvc 파일이 생성됨 (메타데이터)
# 실제 데이터는 .dvcignore에 의해 Git에서 제외됨
```

### 2. 파이프라인 관리 (Pipeline Management)

DVC는 데이터 처리 단계를 파이프라인으로 정의하고 관리합니다:

- **의존성 추적**: 입력 파일, 코드, 파라미터의 변경을 자동 감지
- **단계별 실행**: 변경된 단계와 그 의존성만 재실행
- **재현 가능성**: 동일한 입력에 대해 항상 동일한 결과 보장

```yaml
# dvc.yaml 예시
stages:
  train:
    cmd: python src/train.py
    deps:
      - src/train.py
      - data/raw_data.csv
      - params.yaml
    outs:
      - model/ridge_model.pkl
```

### 3. 실험 관리 (Experiment Tracking)

MLflow와 연동하여 실험을 체계적으로 관리:

- **파라미터 추적**: 하이퍼파라미터와 설정값 기록
- **메트릭 모니터링**: 성능 지표의 변화 추적
- **아티팩트 저장**: 모델 파일과 결과물 보관

## 🏗️ 프로젝트 구조

```
20250709-dvc-app/
├── app.py                 # Flask 웹 애플리케이션
├── params.yaml           # 하이퍼파라미터 설정
├── dvc.yaml             # DVC 파이프라인 정의
├── environment.yml      # Conda 환경 설정
├── data/
│   ├── raw_data.csv     # 원본 데이터 (DVC 추적)
│   └── raw_data.csv.dvc # 데이터 메타데이터
├── src/
│   ├── train.py         # 모델 훈련 스크립트
│   └── evaluate.py      # 모델 평가 스크립트
├── model/               # 훈련된 모델 저장소
├── templates/
│   └── index.html       # 웹 UI 템플릿
└── README.md           # 프로젝트 문서
```

## 🚀 주요 기능

### 1. 다중 모델 지원
- **Linear Regression**: 기본 선형 회귀
- **Ridge Regression**: L2 정규화 적용
- **Lasso Regression**: L1 정규화 적용

### 2. 하이퍼파라미터 관리
```yaml
train:
  alpha: 0.1              # 정규화 강도
  max_iter: 1000          # 최대 반복 횟수
  random_state: 42        # 재현성을 위한 시드
  test_size: 0.2          # 테스트 데이터 비율
  model_type: "ridge"     # 모델 타입 선택
```

### 3. 종합적인 평가 메트릭
- **MSE (Mean Squared Error)**: 평균 제곱 오차
- **R² Score**: 결정 계수
- **MAE (Mean Absolute Error)**: 평균 절대 오차
- **RMSE (Root Mean Squared Error)**: 평균 제곱근 오차

### 4. 웹 기반 실험 관리
- **실시간 상태 모니터링**: Git 커밋, DVC 상태, 파라미터 확인
- **버전 간 전환**: 과거 실험으로 쉽게 롤백
- **MLflow 연동**: 실험 결과 시각화

## 🛠️ 설치 및 설정

### 1. 사전 요구사항
- Python 3.9 이상
- Git
- Conda 또는 Miniconda

### 2. 환경 설정

```bash
# 1. 저장소 클론
git clone <repository-url>
cd 20250709-dvc-app

# 2. Conda 환경 생성 및 활성화
conda env create -f environment.yml
conda activate dvc-mlops-example

# 3. DVC 초기화
dvc init

# 4. 원격 저장소 설정 (선택사항)
dvc remote add -d myremote <remote-storage-url>
```

### 3. 데이터 설정

```bash
# 데이터를 DVC로 추적
dvc add data/raw_data.csv

# 변경사항을 Git에 커밋
git add data/raw_data.csv.dvc
git commit -m "Add initial dataset"
```

## 📖 사용 방법

### 1. 기본 워크플로우

```bash
# 1. 파이프라인 실행
dvc repro

# 2. 결과 확인
dvc status

# 3. 변경사항 커밋
git add .
git commit -m "Update model with new parameters"
```

### 2. 실험 관리

```bash
# 1. 파라미터 수정
# params.yaml 파일에서 하이퍼파라미터 변경

# 2. 실험 실행
dvc repro

# 3. MLflow UI 실행 (별도 터미널)
mlflow ui

# 4. 웹 애플리케이션 실행 (별도 터미널)
python app.py
```

### 3. 버전 관리

```bash
# 1. 현재 상태 확인
git log --oneline
dvc status

# 2. 특정 버전으로 전환
git checkout <commit-hash>
dvc checkout

# 3. 파이프라인 재실행
dvc repro
```

### 4. 웹 인터페이스 사용

1. **애플리케이션 시작**:
   ```bash
   python app.py
   ```

2. **브라우저에서 접속**: `http://localhost:5001`

3. **기능 사용**:
   - 현재 상태 확인
   - Git 커밋 히스토리 탐색
   - 버전 간 전환
   - MLflow UI 연결

## 🔬 실습 시나리오

### 시나리오 1: 하이퍼파라미터 튜닝

1. **초기 실험**:
   ```bash
   # 기본 설정으로 첫 실험
   dvc repro
   git add .
   git commit -m "Initial experiment with ridge regression"
   ```

2. **파라미터 조정**:
   ```yaml
   # params.yaml 수정
   train:
     alpha: 0.5  # 0.1에서 0.5로 변경
     model_type: "lasso"  # ridge에서 lasso로 변경
   ```

3. **재실험**:
   ```bash
   dvc repro
   git add .
   git commit -m "Experiment with lasso regression and higher alpha"
   ```

4. **결과 비교**:
   - MLflow UI에서 두 실험 비교
   - 웹 인터페이스에서 버전 간 전환

### 시나리오 2: 데이터 변경 실험

1. **데이터 수정**:
   ```bash
   # 데이터 파일 수정 후
   dvc add data/raw_data.csv
   git add data/raw_data.csv.dvc
   git commit -m "Update dataset with new features"
   ```

2. **파이프라인 재실행**:
   ```bash
   dvc repro  # 자동으로 모든 단계 재실행
   ```

### 시나리오 3: 실험 롤백

1. **과거 버전 확인**:
   ```bash
   git log --oneline
   ```

2. **특정 버전으로 복원**:
   ```bash
   git checkout <commit-hash>
   dvc checkout
   dvc repro
   ```

## 📊 MLflow 연동

### 실험 추적

MLflow는 다음 정보를 자동으로 추적합니다:

- **파라미터**: 하이퍼파라미터, 모델 설정
- **메트릭**: 성능 지표, 학습 과정
- **아티팩트**: 모델 파일, 시각화 결과

### MLflow UI 사용

```bash
# MLflow UI 시작
mlflow ui

# 브라우저에서 접속: http://127.0.0.1:5000
```

## 🐛 문제 해결

### 일반적인 문제들

1. **DVC 캐시 문제**:
   ```bash
   dvc gc  # 캐시 정리
   dvc checkout  # 파일 복원
   ```

2. **의존성 문제**:
   ```bash
   dvc status  # 상태 확인
   dvc repro --force  # 강제 재실행
   ```

3. **MLflow 연결 문제**:
   ```bash
   # MLflow 서버 재시작
   pkill -f mlflow
   mlflow ui
   ```

## 📈 성능 최적화 팁

1. **DVC 캐시 관리**: 정기적으로 `dvc gc` 실행
2. **병렬 처리**: `dvc repro --jobs 4` 사용
3. **원격 저장소**: 대용량 데이터는 클라우드 저장소 활용

## 🤝 기여하기

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 🙏 감사의 말

- DVC 팀: 훌륭한 데이터 버전 관리 도구 제공
- MLflow 팀: 실험 추적 플랫폼 제공
- Scikit-learn 팀: 머신러닝 라이브러리 제공

---

**Happy Experimenting! 🚀**