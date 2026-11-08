# Give Me Some Credit 과제 수행 및 7분 발표 전략 가이드라인

본 문서는 `Assignment/Assignment.md`의 요구사항과 `Lab 1 ~ Lab 6`의 실습 개념, 그리고 참고 논문 2편(**NOTE**, **NATE**)을 유기적으로 결합하여 과제를 주도적으로 설계·수행할 수 있도록 돕는 **방향성 및 핵심 개념/함수 가이드라인**입니다. 단순 복사-붙여넣기식 코드가 아니라, **각 단계에서 왜(Why) 그 기법을 써야 하는지, 어떤 함수/클래스(What)를 조합해야 하는지, 그리고 어떤 실험과 비교(How)를 통해 논리적인 결론과 7분 발표 스토리를 이끌어낼 수 있는지**에 초점을 맞추었습니다.

---

## 1. 과제 개요 및 데이터셋(`Data/givemesomecredit.csv`) 완벽 이해

### (1) 과제의 핵심 목표
- **문제 정의**: 개인 고객의 재무 및 신용 데이터를 바탕으로 **향후 2년 이내에 심각한 재정적 곤경(90일 이상 연체, 파산 등 채무불이행 = Default)에 빠질 확률을 예측하는 이진 분류(Binary Classification)** 문제.
- **핵심 연구 키워드**: 신용평가(Credit Scoring), 금융 리스크 관리, **불균형 데이터(Imbalanced Class) 분류**, **비선형성(Non-linearity) 포착**, 그리고 **설명 가능한 AI(Explainable AI, XAI)**.
- **최종 산출물**: Lab 1 ~ Lab 6의 학습 내용과 NOTE/NATE 논문의 방법론을 녹여낸 **AI 모델링 및 예측 분석(Predictive Analytics) 결과** 및 **7분 분량의 발표 PPT**.

### (2) 제공된 데이터셋(`Data/givemesomecredit.csv`)의 구조적 특징
과제 데이터는 총 **150,000행 $\times$ 9열**로 구성되어 있으며, 원본 Kaggle 데이터(11열, 결측치 포함)에서 **결측치 처리 및 2개의 핵심 파생 범주형 변수 생성이 완료된 형태(NOTE 논문 Table 1과 동일)**입니다.

| 구분 | 변수명 (Column Name) | 의미 및 설명 (NOTE 논문 기준) | 데이터 타입 및 특성 구분 |
| :--- | :--- | :--- | :--- |
| **Target ($Y$)** | **`SeriousDlqin2yrs`** | 90일 이상 연체 등 심각한 채무불이행 발생 여부 | 이진 분류 타겟 (`0`: 정상 139,974건 [93.3%], `1`: 부도 10,026건 [6.7%], **불균형 비율 $\text{IR} = 13.96$**) |
| **Feature $X_1$** | `RevolvingUtilizationOfUnsecuredLines` | 신용카드 및 개인 신용한도 총액 대비 현재 사용 잔액 비율 | 수치형 (Percentage / 연속형 실수) |
| **Feature $X_2$** | `age` | 대출 신청자의 연령 (세) | 수치형 (Integer / 이산형 정수) |
| **Feature $X_3$** | `NumberOfTime30-59DaysPastDueNotWorse` | 최근 2년간 30~59일 단기 연체 발생 횟수 | 수치형 (Integer) |
| **Feature $X_4$** | `DebtRatio` | 월 총소득 대비 월 부채 상환액, 위자료, 생활비 등의 지출 비율 | 수치형 (Percentage / 연속형 실수) |
| **Feature $X_5$** | `MonthlyIncome` | 월 소득 (결측치 대체 완료된 값 포함) | 수치형 (Real) |
| **Feature $X_6$** | `NumberOfDependents` | 본인을 제외한 부양가족 수 (배우자, 자녀 등) | 수치형/범주형 (Integer / Category) |
| **Feature $X_7$** | `CombinedDefaulted` | 30~89일 또는 90일 이상 중장기 연체 경험 유무를 결합한 파생 변수 | **범주형 (Binary Category: `0` 또는 `1`)** |
| **Feature $X_8$** | `CombinedCreditLoans` | 부동산/주택담보대출을 포함한 개설 대출 및 신용한도 보유 유무 파생 변수 | **범주형 (Binary Category: `0` 또는 `1`)** |

---

## 2. 단계별 과제 수행 로드맵 (Phase 1 ~ Phase 6)

과제를 수행할 때는 Lab 1부터 Lab 6까지의 흐름을 그대로 따라가되, **각 단계마다 NOTE와 NATE 논문의 문제의식을 질문으로 던지고 실험으로 답하는 구조**를 취하는 것이 가장 설득력이 높습니다.

```
[Phase 1] 탐색적 데이터 분석 (EDA) 및 문제 정의 (Lab 1 & Lab 2)
   ↓ "데이터가 얼마나 불균형하며, 왜 선형 모델만으로는 부족한가?"
[Phase 2] 데이터 전처리, 특성 선택 및 비선형 잠재 특성 추출 (Lab 2, Lab 3, Lab 6 + NOTE)
   ↓ "스케일링과 차원 축소(PCA vs Autoencoder)는 어떤 차이를 만드는가?"
[Phase 3] 클래스 불균형 해소(Resampling) 및 검증 프레임워크 설계 (Lab 3, Lab 5 + NATE/NOTE)
   ↓ "데이터 누수(Data Leakage) 없이 불균형(IR=13.96)을 어떻게 해결할 것인가?"
[Phase 4] 알고리즘 스팟 체킹, 트리 앙상블 및 MLP 비교 (Lab 4, Lab 5, Lab 6 + NATE/NOTE)
   ↓ "모수적(Parametric) 모델 vs 비모수적(Non-parametric) 트리 앙상블 vs 신경망(MLP)"
[Phase 5] 하이퍼파라미터 튜닝, 다각도 평가지표 검증 및 모델 저장 (Lab 4, Lab 5 + NATE/NOTE)
   ↓ "Accuracy의 함정을 넘어 AUROC, MCC, F1, 연산 시간으로 최적 모델을 어떻게 확정할까?"
[Phase 6] 설명 가능한 AI (XAI - SHAP) 및 비즈니스 인사이트 도출 (Lab 3 + NOTE/NATE)
   ↓ "블랙박스 모델이 왜 특정 고객을 부도 위험군으로 예측했는가?"
```

---

### Phase 1: 데이터 로드, 기술통계 및 탐색적 시각화 (Lab 1 & Lab 2)

#### 1) 방향성 및 분석 포인트
단순히 그래프를 그리는 데 그치지 말고, **이후 전처리와 모델 선택의 근거(Justification)**를 찾아야 합니다.
- **클래스 불균형 확인**: `SeriousDlqin2yrs`의 `0`과 `1` 비율이 약 `93.3% : 6.7%` ($\text{IR} = 13.961$)임을 수치와 막대그래프(NATE Fig. 1 스타일)로 제시하여 **"아무 처리 없이 학습하면 다수 클래스(0)에 편향될 수밖에 없음"**을 문제로 제기합니다.
- **분포의 왜도(Skewness)와 이상치(Outliers) 진단**: `RevolvingUtilizationOfUnsecuredLines`나 `DebtRatio`는 원래 비율(0~1)이어야 하지만 실제로는 수천~수만에 달하는 극단값(Outlier)이 존재하여 오른쪽 꼬리가 매우 긴 분포(높은 양의 왜도)를 보입니다. 이를 기술통계(`min`, `50%`, `75%`, `max`), 왜도(`skew()`), 상자 그림(Boxplot)으로 보여주고 스케일링이나 트리 모델 도입의 필요성으로 연결합니다.
- **다중공선성 및 비선형성 확인**: 상관행렬(`corr()`)과 PCA 2차원 산점도(NOTE Fig. 1 스타일)를 통해 정상 고객과 부도 고객이 선형 경계 하나로 깔끔하게 분리되지 않고 복잡하게 얽혀 있음(Non-linearly intertwined)을 시각적으로 보여줍니다.

#### 2) 필요한 핵심 개념 및 함수
- **데이터 로드 및 기본 점검 (Lab 1)**:
  - `pandas.read_csv('Data/givemesomecredit.csv')`: 첫 행에 헤더가 포함되어 있으므로 `names` 지정 없이 바로 로드.
  - `df.head()`, `df.shape`, `df.dtypes`, `df.isnull().sum()`: 크기 `(150000, 9)`, 자료형 및 결측치 유무 확인.
  - `df.describe()`, `df.skew()`: 8대 기술통계량 및 왜도 확인 (`pandas.set_option` 활용).
  - `df.groupby('SeriousDlqin2yrs').size()` 또는 `df['SeriousDlqin2yrs'].value_counts(normalize=True)`: 클래스 불균형 비율(IR) 계산.
- **단변량 및 다변량 시각화 (Lab 2)**:
  - 단변량: `df.hist(figsize=...)`, `df.plot(kind='density', subplots=True, layout=(3,3), sharex=False)`, `df.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False)`.
  - 다변량: `df.corr(method='pearson')` + `matplotlib.pyplot.matshow()` (또는 `seaborn.heatmap`), `pandas.plotting.scatter_matrix()` (15만 건 전체를 산점도 행렬로 그리면 느리므로 `df.sample(n=2000, random_state=7)`처럼 층화/무작위 샘플링 후 시각화하는 것이 팁입니다).

---

### Phase 2: 데이터 전처리, 특성 선택 및 잠재 특성 생성 (Lab 2, Lab 3, Lab 6 + NOTE)

#### 1) 방향성 및 분석 포인트
- **스케일링의 선택적 적용 (Lab 2)**:
  - 로지스틱 회귀(LR), LDA, KNN, 신경망(MLP/Autoencoder)은 변수의 스케일 차이(`MonthlyIncome`은 수천~수만, `CombinedDefaulted`는 0~1)에 매우 민감하므로 **`StandardScaler`** 또는 **`MinMaxScaler`**가 필수적입니다.
  - 반면 결정 트리(DT), Random Forest, Gradient Boosting 등 트리 기반 모델은 분할 임계값(Threshold)의 순서만 중요하므로 스케일링 유무에 크게 영향받지 않음을(Lab 2 Challenge Question 원리) 언급하면 깊은 이해도를 보여줄 수 있습니다.
- **특성 중요도 및 선택 (Lab 3)**:
  - 8개 변수가 모두 의미 있는 변수인지 확인하기 위해 `SelectKBest(chi2)`, `RFE`, 그리고 `ExtraTreesClassifier`의 `feature_importances_`를 비교해 봅니다.
- **(심화/가산점 방향) 선형 특성 추출(PCA) vs 비선형 잠재 특성 추출(Autoencoder - NOTE 논문)**:
  - Lab 3에서 배운 `PCA(n_components=2)`로 추출한 2개의 주성분과, Lab 6의 `keras.Sequential`(`Dense` 레이어)을 응용해 만든 간단한 **Stacked Autoencoder(입력 8 $\rightarrow$ 은닉 6 $\rightarrow$ 병목 코딩 층 2 $\rightarrow$ 은닉 6 $\rightarrow$ 복원 출력 8, `loss='mse'`)**의 병목 층(`coding h`)에서 추출한 2개의 비선형 잠재 변수(`en1`, `en2`)를 원본 8개 특성에 이어 붙여(Concatenation, 총 10개 특성) 성능을 비교하면 **NOTE 논문의 핵심 아이디어를 Lab 3 & Lab 6 기술로 직접 구현**한 훌륭한 차별화 포인트가 됩니다!

#### 2) 필요한 핵심 개념 및 함수
- **데이터 스케일링 (`sklearn.preprocessing`, Lab 2)**:
  - `MinMaxScaler(feature_range=(0, 1))`, `StandardScaler()`, `RobustScaler()` (이상치가 많을 때 중앙값과 IQR 기반으로 스케일링하는 대안).
- **특성 선택 및 차원 축소 (`sklearn.feature_selection`, `sklearn.decomposition`, Lab 3)**:
  - `SelectKBest(score_func=chi2, k=...)` (단, `chi2`는 비음수 입력만 허용하므로 `MinMaxScaler` 이후 적용하거나 `f_classif` 사용).
  - `RFE(estimator=LogisticRegression(), n_features_to_select=...)`
  - `ExtraTreesClassifier().fit(X, y).feature_importances_`
  - `PCA(n_components=2)`
- **오토인코더 기반 잠재 변수 추출 (`tensorflow.keras`, Lab 6 + NOTE)**:
  - `Sequential()`, `Dense(..., activation='relu')`, `model.compile(optimizer='adam', loss='mse')` 학습 후 인코더 파트에서 `encoder.predict(X)`로 잠재 벡터 추출 및 `numpy.hstack([X, latent_features])`로 병합.

---

### Phase 3: 클래스 불균형 해소(Resampling) 및 데이터 누수 방지 설계 (Lab 3, Lab 5 + NATE/NOTE)

#### 1) 방향성 및 분석 포인트
이 과제의 **가장 핵심적인 승부처**입니다.
- **절대 주의해야 할 원칙 (Data Leakage 방지 - Lab 5 & NOTE Fig. 5 / NATE Fig. 4)**:
  - 오버샘플링(SMOTE 등)이나 언더샘플링(NearMiss)을 **Train/Test 분할 전에 전체 데이터(150,000건)에 먼저 적용하면 절대 안 됩니다!** 합성된 데이터가 Train과 Test에 동시에 들어가 테스트 성능이 부풀려지는 치명적인 데이터 누수(Data Leakage)가 발생하기 때문입니다.
  - 반드시 **1) 먼저 Train(예: 70% 또는 80%)과 Test(30% 또는 20%)를 층화 분할(`StratifiedKFold` 또는 `train_test_split(..., stratify=y)`)한 뒤, 2) 오직 Train 데이터(`X_train, y_train`)에만 리샘플링을 적용**하고, **Test 데이터(`X_test, y_test`)는 원본 불균형 비율 그대로 유지한 채 평가**해야 합니다.
- **실험 설계 방향 (NATE 논문 벤치마킹)**:
  1. **Baseline (No Resampling)**: 원본 불균형 상태 ($\text{IR} = 13.96$, 부도 약 6.7%).
  2. **Undersampling (`NearMiss`)**: 다수 클래스(정상)를 줄여 균형을 맞춤. 데이터가 약 2만 건(Train 기준 약 1.4만 건)으로 줄어들어 **학습 속도가 비약적으로 빨라지는 효과**와 정보 손실에 따른 성능 변화를 관찰합니다.
  3. **Oversampling (`SMOTE`)**: 소수 클래스(부도)를 KNN 보간으로 증강하여 균형을 맞춤. 정보 손실 없이 데이터를 모두 활용하여 **트리 앙상블 모델에서 최고 성능을 달성하는 과정**을 보여줍니다.
  4. **불균형 비율(`sampling_strategy`) 변화 실험 (선택)**: NATE 논문처럼 소수 클래스 비율을 단계적으로($\text{IR}=6.67 \rightarrow 3.12 \rightarrow 2 \rightarrow 1$) 변화시켰을 때 모델 성능(AUROC, F1, MCC)이 어떻게 변하는지 추세선을 그리면 발표 자료의 완성도가 크게 높아집니다.

#### 2) 필요한 핵심 개념 및 함수
- **검증 분할 (`sklearn.model_selection`, Lab 3 & Lab 6)**:
  - `train_test_split(X, y, test_size=0.3, stratify=y, random_state=7)`
  - `StratifiedKFold(n_splits=5, shuffle=True, random_state=7)` (불균형 데이터이므로 일반 `KFold`보다 클래스 비율을 유지하는 `StratifiedKFold`가 훨씬 적합함).
- **불균형 리샘플링 (`imbalanced-learn` 라이브러리: `imblearn`)**:
  - 오버샘플링: `from imblearn.over_sampling import SMOTE` $\rightarrow$ `SMOTE(sampling_strategy=..., random_state=7).fit_resample(X_train, y_train)`
  - 언더샘플링: `from imblearn.under_sampling import NearMiss` $\rightarrow$ `NearMiss(version=1, sampling_strategy=...).fit_resample(X_train, y_train)`
  - 파이프라인 연동 (교차검증 내 누수 방지, Lab 5 확장): `from imblearn.pipeline import Pipeline as ImbPipeline` (일반 `sklearn.pipeline.Pipeline`은 `fit_resample`을 지원하지 않으므로 교차검증 루프 내부에 리샘플링을 넣을 때는 `imblearn.pipeline.Pipeline`을 사용하거나 `StratifiedKFold`의 `for train_idx, val_idx` 루프 안에서 직접 `fit_resample`을 호출).

---

### Phase 4: 알고리즘 비교 (Spot-Checking) - 모수적 vs 비모수적 앙상블 vs 딥러닝 (Lab 4, Lab 5, Lab 6)

#### 1) 방향성 및 분석 포인트
단순히 "여러 모델을 돌려봤더니 이게 제일 높았다"가 아니라, **모델군(Model Family)별 이론적 특성에 따른 비교 프레임**을 짜야 합니다.
1. **모수적(Parametric) / 선형 모델군 (벤치마크 기준선)**:
   - `LogisticRegression` (금융권 전통적 표준 모델), `LinearDiscriminantAnalysis (LDA)`, `GaussianNB`.
   - 해석은 직관적이나 GMSC 데이터의 비선형적 상호작용을 포착하는 데 한계가 있음을 실험으로 확인합니다.
2. **비모수적(Non-parametric) 단일 모델군**:
   - `KNeighborsClassifier (KNN)`, `DecisionTreeClassifier (CART)`.
   - 단일 결정 트리는 가지치기(Pruning)를 하지 않으면 오버샘플링된 데이터에서 심각한 과적합을 일으킴을 확인합니다.
3. **비모수적(Non-parametric) 트리 기반 앙상블 모델군 (핵심 주력 모델 - Lab 5 & NOTE/NATE)**:
   - **Bagging 계열**: `RandomForestClassifier` (NOTE 최고 성능 모델), `ExtraTreesClassifier`, `BaggingClassifier`.
   - **Boosting 계열**: `GradientBoostingClassifier` (NATE 최고 성능 모델), `AdaBoostClassifier`, 그리고 `XGBClassifier` (`xgboost`).
   - **Voting 계열**: 상위 성능 모델들을 묶은 `VotingClassifier(voting='soft')`.
4. **딥러닝 신경망 모델군 (Lab 6)**:
   - `tf.keras.Sequential` 기반의 **MLP (Multi-layered Perceptron)** (예: 입력 8 $\rightarrow$ Dense 32/16 `relu` + `Dropout` $\rightarrow$ 출력 1 `sigmoid`, `binary_crossentropy`, `adam`).

#### 2) 필요한 핵심 개념 및 함수
- **기본 분류기 (Lab 4)**: `LogisticRegression`, `LinearDiscriminantAnalysis`, `GaussianNB`, `KNeighborsClassifier`, `DecisionTreeClassifier`, `SVC` (단, 기본 RBF 커널 `SVC()`는 15만 건 데이터에서 학습 복잡도가 $O(N^2 \sim N^3)$이라 수 시간 이상 걸릴 수 있으므로, 사용 시 `LinearSVC`를 쓰거나 서브샘플링/NearMiss 데이터에서만 비교하는 것이 현명합니다).
- **앙상블 분류기 (Lab 5 & 논문)**: `BaggingClassifier`, `RandomForestClassifier`, `ExtraTreesClassifier`, `AdaBoostClassifier`, `GradientBoostingClassifier`, `VotingClassifier`, `xgboost.XGBClassifier`.
- **알고리즘 비교 시각화 (Lab 4)**: `cross_val_score` 결과 리스트를 `pyplot.boxplot(results)`로 그려 폴드별 평균 성능과 분산(모델 안정성)을 한눈에 비교.

---

### Phase 5: 하이퍼파라미터 튜닝, 다각도 평가지표 검증 및 모델 저장 (Lab 4, Lab 5 + NATE/NOTE)

#### 1) 방향성 및 분석 포인트
- **하이퍼파라미터 튜닝과 트리 가지치기(Pruning)**:
  - NOTE와 NATE 논문 모두 강조하듯, 트리 모델이 합성 데이터(SMOTE/cWGAN)의 노이즈에 과적합되지 않도록 `max_depth`, `min_samples_split`, `min_samples_leaf`, `n_estimators`, `learning_rate`를 `GridSearchCV` 또는 `RandomizedSearchCV`로 튜닝합니다.
  - **실전 팁**: 15만~28만 건 데이터에 대해 `GridSearchCV`로 수십 개 조합을 전부 탐색하면 CPU에서 수 시간이 걸립니다. 따라서 Lab 6의 조언처럼 **층화 추출된 일부 샘플이나 `RandomizedSearchCV(n_iter=10~20, n_jobs=-1)`를 활용해 빠르게 최적 영역을 찾은 뒤 전체 학습에 적용**하는 전략이 매우 효율적입니다.
- **불균형 데이터에 적합한 평가 지표 종합 비교 (Lab 4 + NATE)**:
  - **Accuracy의 한계 증명**: 원본 데이터에서 다수 클래스만 찍어도 Accuracy가 93.3%가 나오는 현상을 표로 제시합니다.
  - **핵심 평가 지표**:
    1. **AUROC (`roc_auc_score`, `roc_curve`)**: 목표치 — NATE 수준(`0.96+`) 또는 NOTE 수준(`0.98+`, 단 교차검증 설계에 따라 Test셋 기준 `0.86+` 수준인지 학습/리샘플링 결합 기준인지 명확히 구분하여 해석).
    2. **F1-Score & Classification Report**: 부도 클래스(`1`)에 대한 Precision, Recall, F1 변화 확인.
    3. **MCC (`matthews_corrcoef`)**: NATE 논문에서 사용한 불균형 데이터 표준 지표.
    4. **Confusion Matrix & Type I / Type II Error 분석 (Lab 4 Question (i), (j))**: 신용평가에서는 **실제 부도낼 고객(1)을 정상(0)으로 잘못 승인해 주는 제2종 오류(Type II Error, False Negative)의 재무적 손실이 훨씬 치명적**입니다. 리샘플링 전후로 False Negative가 얼마나 극적으로 줄어들었는지 혼동 행렬로 비교하면 발표에서 매우 강력한 인상을 줍니다!
    5. **학습 소요 시간 (`time.time()`) 비교**: NATE Table 8처럼 `SMOTE + GB`의 높은 정확도와 `NearMiss + XGBoost`의 압도적인 속도(수십 배 차이)를 비교하여 실무적 의사결정 기준을 제시합니다.
- **최종 모델 저장 (Lab 5)**:
  - 최종 선정된 최고 성능 파이프라인/모델을 `joblib.dump()` 또는 `pickle.dump()`로 직렬화하여 저장합니다.

#### 2) 필요한 핵심 개념 및 함수
- **튜닝 (`sklearn.model_selection`, Lab 5)**: `GridSearchCV(estimator, param_grid, scoring='roc_auc', cv=5, n_jobs=-1)`, `RandomizedSearchCV`. (논문 탐색 공간 참고: `max_depth: [1, 20]`, `min_samples_split: [2, 5, 10]`, `min_samples_leaf: [1, 2, 4, 8]`, `n_estimators: [100, 1000]`).
- **평가지표 (`sklearn.metrics`, Lab 4 & NATE)**:
  - `roc_auc_score(y_test, y_prob)`, `roc_curve(y_test, y_prob)`
  - `confusion_matrix(y_test, y_pred)`
  - `classification_report(y_test, y_pred)`
  - `f1_score(y_test, y_pred)`
  - `matthews_corrcoef(y_test, y_pred)`
  - `log_loss(y_test, y_prob)`
- **통계적 유의성 검정 (NOTE Table 5)**: `scipy.stats.friedmanchisquare(...)` (여러 리샘플링/모델 간 폴드별 성능 차이가 우연이 아님을 $p < 0.05$로 검증).
- **모델 저장 (Lab 5)**: `joblib.dump(best_model, 'best_credit_model.sav')`, `joblib.load(...)`.

---

### Phase 6: 설명 가능한 AI (XAI - SHAP)를 통한 모델 해석 (NOTE & NATE 핵심)

#### 1) 방향성 및 분석 포인트
비모수 트리 앙상블 모델(Random Forest, Gradient Boosting, XGBoost)의 유일한 약점인 **'블랙박스(해석 불가능성)'** 문제를 해결하여 금융권 규제 및 실무 적용 가능성을 입증하는 마무리 단계입니다.
- **Lab 3의 `feature_importances_` vs 논문의 `SHAP` 비교**:
  - Lab 3에서 배운 트리의 불순도 기반 `feature_importances_`는 전체 데이터에서 어떤 변수가 자주 쓰였는지만 알려줄 뿐, **그 변수가 부도 확률을 높이는 방향(+)으로 작용했는지 낮추는 방향(-)으로 작용했는지, 그리고 특정 개인 고객에게는 왜 그런 판정이 나왔는지(Local explanation)** 설명하지 못합니다.
  - 반면 **`shap.TreeExplainer`**를 사용하면:
    1. **전역적 해석 (Global)**: **Summary Bar Plot & Violin/Beeswarm Plot**을 통해 전체 고객에 대해 `CombinedDefaulted`, `RevolvingUtilizationOfUnsecuredLines`, `NumberOfTime30-59DaysPastDueNotWorse` 순으로 영향력이 크며, 값의 고저(색상)에 따라 부도 위험이 어떻게 변하는지 한눈에 보여줍니다.
    2. **국소적 해석 (Local)**: 테스트셋에서 **실제 부도로 예측된 고위험 고객 1명**과 **정상으로 예측된 고객 1명(또는 오분류된 경계 고객 1명)**을 뽑아 **Force Plot / Waterfall Plot / Decision Plot**으로 각 변수의 기여도($+\phi_i, -\phi_i$)를 분석합니다.

#### 2) 필요한 핵심 개념 및 함수
- **`shap` 라이브러리**:
  - `explainer = shap.TreeExplainer(best_tree_model)` (로지스틱 회귀 비교 시 `shap.LinearExplainer`)
  - `shap_values = explainer.shap_values(X_test_sample)` (또는 `explainer(X_test_sample)`)
  - 전역 시각화: `shap.summary_plot(shap_values, X_test_sample, plot_type='bar')`, `shap.summary_plot(shap_values, X_test_sample, plot_type='violin')`
  - 국소 시각화: `shap.force_plot(...)`, `shap.decision_plot(...)`, `shap.plots.waterfall(...)`

---

## 3. 7분 발표용 PPT 구성 가이드라인 (권장 10~12 슬라이드)

7분 발표는 **슬라이드당 약 35~40초**, 총 **10~12장 내외**로 구성하는 것이 시간 관리에 가장 이상적입니다.

| 슬라이드 번호 | 섹션 제목 | 핵심 포함 내용 (시각 자료 중심) | 권장 소요 시간 |
| :---: | :--- | :--- | :---: |
| **Slide 1** | **표지 (Title)** | 발표 제목(예: *불균형 신용평가 데이터를 위한 비모수 앙상블 모델링 및 XAI 분석*), 과목명, 이름/학번 | 15초 |
| **Slide 2** | **배경 및 문제 정의 (Problem Statement)** | - 신용평가(Credit Scoring)에서 부도 예측의 중요성<br>- 3대 핵심 과제: ① 심각한 클래스 불균형($\text{IR}=13.96$), ② 비선형 데이터 구조, ③ 고성능 AI 모델의 설명 가능성(XAI) 확보 (NOTE & NATE 논문 배경 연결) | 40초 |
| **Slide 3** | **데이터 탐색 및 기술통계 (EDA - Lab 1 & 2)** | - GMSC 데이터 구조 (150,000건 $\times$ 9열: 타겟 1 + 수치형/파생 범주형 8개 변수 표)<br>- 타겟 클래스 불균형 분포 그래프(93.3% vs 6.7%) 및 주요 변수의 왜도(Skewness)/박스플롯/상관행렬 요약 | 45초 |
| **Slide 4** | **전체 분석 파이프라인 아키텍처 (Methodology)** | - 전처리(`StandardScaler` / `PCA` / `NSA`) $\rightarrow$ Train/Test 분할 $\rightarrow$ 리샘플링(`SMOTE` vs `NearMiss`) $\rightarrow$ 모델링(모수 vs 비모수 앙상블/MLP) $\rightarrow$ 튜닝(`GridSearchCV`) $\rightarrow$ `SHAP` 해석으로 이어지는 **데이터 누수 없는 전체 시스템 다이어그램** (NOTE Fig. 5 / NATE Fig. 4 스타일) | 45초 |
| **Slide 5** | **Baseline 실험 및 평가지표 선정 이유 (Lab 4)** | - 원본 불균형 데이터에서 6~8개 기본 알고리즘 Spot-Checking 결과<br>- **왜 Accuracy(93.2%)는 무의미한가?** (F1-score `0.05`의 충격 제시)<br>- 대안 지표 정의: **AUROC, F1-Score, MCC, 그리고 Type II Error (FN)** | 40초 |
| **Slide 6** | **핵심 실험 1: 리샘플링(SMOTE vs NearMiss) 및 IR 변화 효과 (NATE 적용)** | - 불균형 비율(IR) 조정에 따른 모수 모델(LR) vs 비모수 트리 앙상블(RF, GB, XGB)의 AUROC / F1 / MCC 상승 그래프<br>- 혼동 행렬(Confusion Matrix) 비교를 통한 부도 고객 탐지율(Recall / Type II Error 감소) 개선 효과 | 50초 |
| **Slide 7** | **핵심 실험 2: 특성 추출(PCA vs NSA) 및 딥러닝(MLP) 비교 (Lab 3, 6 + NOTE 적용)** | - 선형 차원 축소(PCA) 대비 비선형 오토인코더(NSA) 잠재 특성 결합 시의 성능 변화<br>- Keras MLP 모델과 트리 앙상블 모델의 성능 비교 및 하이퍼파라미터 튜닝(`GridSearchCV` + Pruning) 결과 요약 표 | 50초 |
| **Slide 8** | **모델 종합 비교: 정확도 vs 연산 효율성 트레이드오프** | - 전체 모델 최종 성적표 (AUROC, MCC, F1, **학습 소요 시간**)<br>- 최고 정확도 모델(예: Oversampling + RF/GB) vs 최고 가성비/실시간성 모델(예: NearMiss + XGBoost)의 실무 적용 시나리오 비교 | 40초 |
| **Slide 9** | **XAI 분석: 모델은 무엇을 보고 부도를 예측하는가? (SHAP)** | - **Global 해석**: SHAP Bar Plot & Violin Plot (`CombinedDefaulted`, `RevolvingUtilization`, `30-59DaysPastDue` 상위 3대 요인 도출)<br>- **Local 해석**: 실제 정상 판정 고객(예: $P=0.22$) vs 부도 위험 고객(예: $P=0.76$)의 Force/Decision Plot 비교를 통한 맞춤형 심사 근거 제시 | 55초 |
| **Slide 10** | **결론 및 향후 과제 (Conclusion & Future Work)** | - 핵심 요약 3줄 (비모수 앙상블+오버샘플링의 우수성, 하이퍼파라미터 튜닝을 통한 과적합 제어, SHAP을 통한 투명한 금융 의사결정 달성)<br>- 한계점 및 향후 연구 방향 (NSA와 cWGAN의 통합 학습, VAE 도입 등 NOTE/NATE Discussion 내용 반영) | 30초 |
| *(총합)* | | **10장 구성, 약 6분 50초 분량 (질의응답 대비 완벽한 타임라인)** | **약 6분 50초** |

