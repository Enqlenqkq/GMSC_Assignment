# Algorithm Analysis 과제 및 실습 통합 문서 (`docs/`)

본 디렉토리는 `goal.txt`에 정의된 4가지 핵심 요청사항을 수행하기 위해 작성된 종합 정리 및 가이드라인 문서들로 구성되어 있습니다.

---

## 문서 목록 및 구성

1. **[01_Lab1_to_Lab7_Summary.md](./01_Lab1_to_Lab7_Summary.md)**
   - **목적**: `Lab/Lab1 ~ Lab7` 실습 자료의 상세 개념, 워크플로우, 핵심 함수/클래스 및 주요 확인 질문(`Verify your understanding`) 총정리 요약본
   - **포함 내용**:
     - **Lab 1**: CSV 데이터 로드 3가지 방식(`csv.reader`, `numpy.loadtxt`, `pandas.read_csv`) 및 8대 기술통계량/상관관계/왜도 분석
     - **Lab 2**: 단변량(히스토그램, 밀도 플롯, 박스플롯)/다변량(상관행렬, 산점도 행렬) 시각화 및 4대 데이터 전처리(`MinMaxScaler`, `StandardScaler`, `Normalizer`, `Binarizer`)
     - **Lab 3**: 4대 특성 선택/차원 축소(`SelectKBest`, `RFE`, `PCA`, `ExtraTrees` 중요도) 및 4대 검증 리샘플링(`train_test_split`, `KFold`, `LeaveOneOut`, `ShuffleSplit`)
     - **Lab 4**: 5대 분류 평가지표(`Accuracy`, `LogLoss`, `AUROC`, `Confusion Matrix` & Type I/II 오류, `Classification Report`) 및 6대 ML 알고리즘 스팟 체킹
     - **Lab 5**: 데이터 누수 방지를 위한 `Pipeline` & `FeatureUnion`, 3대 앙상블(Bagging, Boosting, Voting), 하이퍼파라미터 튜닝(`GridSearchCV`, `RandomizedSearchCV`), 모델 직렬화(`pickle`, `joblib`)
     - **Lab 6**: Keras `Sequential` 기반 다층 퍼셉트론(MLP), 검증 데이터 분할 3방식, `KerasClassifier`를 통한 Scikit-Learn 교차검증 및 그리드 서치 연동
     - **Lab 7**: Keras 기반 합성곱 신경망(CNN, `Conv2D`, `MaxPooling2D`, `Dropout`, `Flatten`), 학습 곡선(`history`) 분석 및 오분류 샘플 시각화

2. **[02_Paper_Summary_NOTE_NATE.md](./02_Paper_Summary_NOTE_NATE.md)**
   - **목적**: `Paper/` 내 참고 논문 2편(**NOTE** 및 **NATE**) 심층 요약 및 과제 적용 전략 정리
   - **포함 내용**:
     - **NOTE** (*Nature Scientific Reports*, 2024): 비모수 적층 오토인코더(NSA)를 통한 비선형 잠재 특성 추출 + `cWGAN-GP` 오버샘플링 + 트리 기반 분류(Random Forest **AUROC 0.9837**) + Friedman 검정 + SHAP 해석
     - **NATE** (*PLOS ONE*, 2024): 불균형 비율(IR) 단계별 `SMOTE`(오버샘플링) vs `NearMiss`(언더샘플링) 비교 + 모수(LR) vs 비모수 트리 앙상블(RF, GB, XGB) 비교(GB+SMOTE **AUROC 0.9649, MCC 0.8104, F1 0.9072**) + 정확도와 연산 시간의 트레이드오프 + SHAP 전역/국소 해석
     - 과제 데이터(`Data/givemesomecredit.csv`, 150,000행 $\times$ 9열)와 두 논문 간의 변수 매핑 및 벤치마크 활용 전략

3. **[03_Assignment_Guideline.md](./03_Assignment_Guideline.md)**
   - **목적**: `Assignment/Assignment.md` 및 `Data/givemesomecredit.csv` 기반 과제 수행 방향성, 단계별 필수 개념/함수 매핑, 그리고 7분 발표 PPT 구성 가이드라인
   - **포함 내용**:
     - GMSC 데이터셋 9개 변수(타겟 1 + 수치형/파생 범주형 8) 완벽 해설
     - **Phase 1 ~ Phase 6** 단계별 분석 설계 방향, 핵심 질문, 주의사항(데이터 누수 방지 등) 및 필수 파이썬 함수/클래스 정리
     - **7분 발표용 PPT 10장 슬라이드별 스토리라인 및 시간 배분표**

4. **[04_GPU_Acceleration_Guide.md](./04_GPU_Acceleration_Guide.md)**
   - **목적**: 로컬 GPU(NVIDIA CUDA) 및 클라우드 GPU를 활용한 머신러닝/딥러닝 학습 가속 방법 정리
   - **포함 내용**:
     - 알고리즘별 CPU vs GPU 가속 지원 현황 비교표
     - `XGBClassifier(tree_method='hist', device='cuda')` 및 `XGBRFClassifier`를 통한 트리 앙상블 GPU 초고속 학습법
     - Scikit-Learn 기본 모델의 멀티코어 병렬화(`n_jobs=-1`) 및 `HistGradientBoostingClassifier` 활용법
     - Windows 환경에서의 Keras / TensorFlow / PyTorch GPU 활성화 3가지 방법 및 대용량 데이터 배치 크기(`batch_size`) 최적화 팁

