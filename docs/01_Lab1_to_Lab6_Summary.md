# Lab 1 ~ Lab 6 핵심 개념 및 함수 총정리 요약본

본 문서는 `Lab/Lab1`부터 `Lab/Lab6`까지의 실습 자료에 등장하는 머신러닝 및 딥러닝 핵심 개념, 전체 워크플로우, 주요 파이썬 라이브러리/클래스/함수, 그리고 각 실습의 핵심 질문(`Verify your understanding`)을 통해 얻어야 할 시사점을 체계적으로 정리한 요약본입니다.

---

## 한눈에 보는 Lab 1 ~ Lab 6 전체 흐름

```
[Lab 1] 데이터 로드 및 기술통계 (Data Loading & Descriptive Statistics)
   ↓
[Lab 2] 데이터 시각화 및 전처리/스케일링 (Visualisation & Data Preparation)
   ↓
[Lab 3] 특성 선택/차원 축소 및 리샘플링/교차검증 (Feature Selection & Resampling)
   ↓
[Lab 4] 분류 평가 지표 및 ML 알고리즘 비교 (Evaluation Metrics & Spot-Checking)
   ↓
[Lab 5] 파이프라인 자동화, 앙상블, 하이퍼파라미터 튜닝, 모델 저장 (Automation, Ensembles, Tuning & Saving)
   ↓
[Lab 6] Keras 기반 다층 퍼셉트론(MLP)과 Scikit-Learn 연동 (Deep Learning with Keras)
```

---

## Lab 1: 머신러닝 데이터 로드 및 기술통계 (Loading ML Data and Descriptive Statistics)

### 1. CSV 데이터 로드 시 고려사항 (RFC 4180 기준)

머신러닝 프로젝트의 첫 단계는 원본 데이터를 정확히 메모리에 적재하는 것입니다. CSV 파일 로드 시 다음 4가지를 반드시 확인해야 합니다.

- **File Header (파일 헤더)**: 첫 번째 행에 열 이름(속성명)이 포함되어 있는지 여부. 헤더가 없다면(`pima-indians-diabetes.csv`처럼) 수동으로 `names` 리스트를 지정해야 하며, 헤더가 있다면(`givemesomecredit.csv`처럼) 자동으로 컬럼명으로 인식하도록 설정합니다.
- **Comments (주석)**: `#` 등으로 시작하는 주석 행이 존재하는지 확인하고 필요시 주석 문자를 지정합니다.
- **Delimiter (구분자)**: 기본값은 쉼표(`,`)이나 탭(`\t`)이나 공백이 쓰일 경우 명시적으로 지정해야 합니다.
- **Quotes (따옴표)**: 공백이 포함된 문자열 필드를 감싸는 따옴표 문자(기본값 `"`).

### 2. 파이썬에서 CSV 파일을 로드하는 3가지 방법 비교

| 방법                          | 사용 모듈 및 함수                                                         | 반환 타입                                 | 특징 및 권장 용도                                                                                              |
| :---------------------------- | :------------------------------------------------------------------------ | :---------------------------------------- | :------------------------------------------------------------------------------------------------------------- |
| **1. Python 표준 라이브러리** | `csv.reader(open(filename, 'rt'), delimiter=',', quoting=csv.QUOTE_NONE)` | `list` → `numpy.array(x).astype('float')` | 가장 저수준(low-level) 방식. 기본적으로 모든 값을 문자열로 읽으므로 수동 형변환(`.astype('float')`) 필요       |
| **2. NumPy 활용**             | `numpy.loadtxt(raw_data, delimiter=",")`                                  | `numpy.ndarray`                           | 헤더가 없고 모든 데이터가 동일한 수치형 포맷일 때 빠르고 간편함 (`urllib.urlopen`과 결합해 URL 로드 가능)      |
| **3. Pandas 활용 (권장)**     | `pandas.read_csv(filename, names=names)`                                  | `pandas.DataFrame`                        | **가장 유연하고 권장되는 방식**. 헤더 유무, 다양한 자료형 혼합, 결측치 처리, 즉각적인 요약 통계 및 시각화 지원 |

> **참고 (`open()` 함수의 파일 모드)**: `'r'`(읽기 기본), `'w'`(쓰기, 기존 내용 삭제), `'x'`(배타적 생성), `'a'`(이어쓰기), `'b'`(바이너리 모드), `'t'`(텍스트 모드 기본), `'+'`(읽기/쓰기 갱신).

### 3. 기술통계 (Descriptive Statistics)를 통한 데이터 탐색 (EDA)

원본 데이터를 직접 들여다보는 것은 전처리 및 모델링 아이디어를 얻는 가장 확실한 방법입니다.

| 탐색 목적                        | Pandas DataFrame 속성/메서드                           | 주요 확인 포인트 및 의미                                                                                                                                                                                                                |
| :------------------------------- | :----------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **상위 데이터 직접 확인**        | `data.head(20)`                                        | 처음 $N$개 행을 출력하여 값의 형태, 이상 징후(예: 0이 나올 수 없는 변수에 0 존재 여부) 파악                                                                                                                                             |
| **데이터 차원(크기) 확인**       | `data.shape`                                           | `(행 개수, 열 개수)` 튜플 반환. 행이 너무 많으면 학습 시간이 길어지고, 너무 적으면 과소적합 위험. 열(특성)이 너무 많으면 **차원의 저주(Curse of Dimensionality)** 발생 가능                                                             |
| **속성별 자료형 확인**           | `data.dtypes`                                          | 각 컬럼이 정수형(`int64`), 실수형(`float64`), 문자열/범주형(`object`)인지 확인하여 인코딩/형변환 필요성 판단                                                                                                                            |
| **8대 요약 통계량 확인**         | `data.describe()`                                      | `count`(유효 개수 → 결측치 확인), `mean`(평균), `std`(표준편차), `min`(최솟값), `25%`, `50%`(중앙값/Median), `75%`, `max`(최댓값) 출력 (`pd.set_option('display.width', 100)`, `pd.set_option('precision', 3)` 등으로 출력 가독성 조절) |
| **클래스 분포(불균형) 확인**     | `data.groupby('class').size()` (또는 `value_counts()`) | 분류(Classification) 문제에서 타겟 클래스 간 비율 확인. **불균형(Imbalanced) 데이터**일 경우 별도의 리샘플링이나 평가 지표 설계가 필수적임                                                                                              |
| **변수 간 상관관계 분석**        | `data.corr(method='pearson')`                          | 두 변수 간 선형 관계($-1 \sim +1$, 피어슨 상관계수) 행렬 계산. **선형 회귀(Linear Regression)나 로지스틱 회귀(Logistic Regression)**는 입력 변수 간 다중공선성(높은 상관관계)이 있으면 성능이 저하되므로 확인 필요                      |
| **단변량 분포의 왜도(Skewness)** | `data.skew()`                                          | 정규분포(Gaussian) 대비 치우친 정도 측정. 양수(+)면 오른쪽 꼬리가 긴 분포(Right/Positive skew), 음수(-)면 왼쪽 꼬리가 긴 분포(Left/Negative skew), 0에 가까울수록 대칭적 정규분포                                                       |

---

## Lab 2: 데이터 시각화 및 데이터 전처리 (Preparing Data)

### 1. 데이터 시각화 (Visualisation)

데이터의 분포, 이상치(Outliers), 변수 간 상호작용을 가장 빠르게 파악하는 도구입니다.

#### (1) 단변량 플롯 (Univariate Plots): 각 속성을 독립적으로 분석

1. **히스토그램 (Histograms)**: `data.hist()`
   - 데이터를 구간(bins, 기본값 `bins=10`)으로 나누어 관측치 빈도를 막대로 표시.
   - 분포 형태(정규분포, 왜도, 지수분포 등)와 이상치 존재 여부를 직관적으로 확인 (`figsize` 인자로 크기 조절 가능).
2. **밀도 플롯 (Density Plots / KDE)**: `data.plot(kind='density', subplots=True, layout=(3,3), sharex=False)`
   - 히스토그램의 상단을 매끄러운 곡선으로 추상화하여 각 변수의 연속적인 확률밀도 분포 형태를 더 명확히 확인.
3. **상자 수염 그림 (Box and Whisker Plots / Boxplots)**: `data.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False)`
   - 중앙값(Median, 50% 선), 사분위수 범위(IQR: 25% ~ 75% 상자), 수염(Whiskers: 데이터의 전체적 퍼짐 정도)을 요약.
   - 수염 바깥(IQR의 $1.5$배 초과 영역)에 찍히는 점들을 통해 **이상치(Outlier) 후보**를 명확히 식별.

#### (2) 다변량 플롯 (Multivariate Plots): 여러 변수 간의 상호작용 분석

1. **상관행렬 플롯 (Correlation Matrix Plot)**:
   - `correlations = data.corr()` 결과를 `ax.matshow(correlations, vmin=-1, vmax=1)` 및 `fig.colorbar(cax)`로 히트맵 형태로 시각화.
   - 축에 변수명(`set_xticklabels`, `set_yticklabels`)을 지정하면 어떤 변수 쌍이 강하게 상관되어 있는지 한눈에 파악 가능하며, 대각선은 자기 자신과의 상관계수(1.0)를 의미하는 대칭 행렬임.
2. **산점도 행렬 (Scatter Plot Matrix)**:
   - `from pandas.plotting import scatter_matrix` → `scatter_matrix(data, figsize=[20, 20])`
   - 모든 변수 쌍에 대해 2차원 산점도를 그려 변수 간 구조적/선형적 관계나 군집 패턴을 포착. 대각선에는 자기 자신과의 산점도 대신 각 변수의 히스토그램이 표시됨.

### 2. 데이터 변환 및 전처리 (Data Preparation & Preprocessing)

알고리즘마다 데이터에 대해 가정하는 분포나 스케일이 다르므로 적절한 변환이 필요합니다.

#### Scikit-Learn 전처리의 2가지 표준 방식

- **Fit and Multiple Transform (권장)**: `scaler = StandardScaler().fit(X_train)`으로 학습 데이터에서 파라미터(평균, 표준편차 등)를 먼저 추정한 뒤, `scaler.transform(X_train)` 및 `scaler.transform(X_test)`로 학습/검증/테스트 데이터에 동일하게 적용. (데이터 누수 방지!)
- **Combined Fit-And-Transform**: `scaler.fit_transform(X)`으로 일회성 요약이나 시각화 시 편리하게 사용.

#### 4가지 핵심 전처리(스케일링) 기법 비교 (`sklearn.preprocessing`)

| 전처리 기법                                     | 클래스 및 주요 파라미터              | 변환 원리 및 수식적 의미                                                           | 적합한 알고리즘 및 활용 상황                                                                             |
| :---------------------------------------------- | :----------------------------------- | :--------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------- |
| **1. Rescale Data (정규화/최소-최대 스케일링)** | `MinMaxScaler(feature_range=(0, 1))` | 각 열(특성)의 최솟값을 0, 최댓값을 1로 맞추어 $[0, 1]$ 구간으로 스케일 조정        | 경사하강법(Gradient Descent) 기반 최적화, 회귀, **신경망(Neural Networks)**, 거리 기반 알고리즘(**KNN**) |
| **2. Standardise Data (표준화)**                | `StandardScaler()`                   | 각 열(특성)의 평균(Mean)을 $0$, 표준편차(Std)를 $1$인 표준정규분포로 변환          | 입력 변수가 가우시안(정규) 분포를 따른다고 가정하는 **선형 회귀, 로지스틱 회귀(LR), 선형판별분석(LDA)**  |
| **3. Normalise Data (단위 노름 정규화)**        | `Normalizer()`                       | 열(Column)이 아닌 **각 행(관측치 샘플 벡터)의 길이(L2 Norm 등)가 1이 되도록** 조정 | 0이 많은 **희소 데이터(Sparse dataset)**에서 신경망이나 거리 기반 알고리즘(KNN) 적용 시                  |
| **4. Binarise Data (이진화)**                   | `Binarizer(threshold=0.0)`           | 지정한 임계값(`threshold`)보다 크면 $1$, 같거나 작으면 $0$으로 변환                | 확률값을 확정적(crisp) 이진 값으로 만들거나, 의미 있는 이진 파생 변수(Feature Engineering)를 생성할 때   |

> **핵심 시사점 (Challenge Question (o))**: 의사결정나무(`DecisionTreeClassifier`)와 같은 트리 기반 모델은 데이터의 분할 기준점(Threshold)을 순서 기반으로 찾으므로, 행 단위 정규화(`Normalizer`) 등을 무작정 적용하면 오히려 원본 데이터(Raw Data)보다 정확도가 떨어질 수 있습니다(예: $0.687 \rightarrow 0.638$). 즉, **전처리가 항상 성능을 높이는 것은 아니며, 모델의 특성(선형/거리 기반 vs 트리 기반)에 맞는 전처리를 선택해야 합니다.**

---

## Lab 3: 특성 선택(Feature Selection), 차원 축소 및 리샘플링(Resampling)

### 1. 특성 선택 (Feature Selection) 및 차원 축소

불필요하거나 중복된(redundant) 특성은 모델 성능(특히 선형/로지스틱 회귀)을 저하시킵니다.

- **특성 선택의 3대 이점**:
  1. **과적합 감소 (Reduces Overfitting)**: 노이즈에 기반한 의사결정 가능성을 줄임.
  2. **정확도 향상 (Improves Accuracy)**: 오도하는(misleading) 데이터를 제거하여 예측력 향상.
  3. **학습 시간 단축 (Reduces Training Time)**: 데이터 차원이 줄어들어 알고리즘 학습 속도 향상.

#### 4가지 특성 선택 및 추출 기법 비교

| 기법 분류                                         | Scikit-Learn 클래스 및 함수                                                   | 동작 원리                                                                                                                                  | 주요 확인 속성                                                                       |
| :------------------------------------------------ | :---------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------- |
| **1. 단변량 통계 선택 (Univariate Selection)**    | `SelectKBest(score_func=chi2, k=4)` (`sklearn.feature_selection`)             | 카이제곱($\chi^2$, 비음수 특성 대상) 등 통계적 검정을 통해 타겟 변수와 가장 연관성이 높은 상위 $k$개 특성 선택                             | `fit.scores_` (각 특성의 통계 점수), `fit.transform(X)`                              |
| **2. 재귀적 특성 제거 (RFE)**                     | `RFE(estimator=model, n_features_to_select=3)` (`sklearn.feature_selection`)  | 기저 모델(예: `LogisticRegression`)을 반복 학습시키면서 가장 중요도가 낮은 특성을 하나씩 제거하여 상위 $k$개 특성 조합 탐색                | `fit.n_features_`, `fit.support_` (선택 여부 `True/False`), `fit.ranking_` (1순위=1) |
| **3. 주성분 분석 (PCA - 차원 축소)**              | `PCA(n_components=3)` (`sklearn.decomposition`)                               | 선형대수(고유값 분해/특이값 분해)를 이용해 원본 특성들을 서로 직교하는 $k$개의 주성분(Principal Components)으로 압축 변환 (선형 특성 추출) | `fit.explained_variance_ratio_` (각 주성분의 설명 분산 비율), `fit.components_`      |
| **4. 트리 기반 특성 중요도 (Feature Importance)** | `ExtraTreesClassifier()` 또는 `RandomForestClassifier()` (`sklearn.ensemble`) | 배깅 기반 결정 트리 앙상블을 학습시킨 후, 불순도(Gini impurity / Information Gain) 감소량 평균으로 각 특성의 중요도 산출                   | `model.feature_importances_` (값이 클수록 중요한 변수)                               |

### 2. 리샘플링 / 모델 검증 기법 (Resampling Methods for Model Evaluation)

> **주의**: 여기서 Lab 3의 'Resampling'은 불균형 클래스 오버샘플링이 아니라, **보지 못한 새로운 데이터(Unseen data)에 대한 모델의 일반화 성능을 통계적으로 추정하기 위한 데이터 분할/교차검증 기법**을 의미합니다. 학습 데이터로만 평가하면 데이터를 단순 암기하는 **과적합(Overfitting)** 문제를 감지할 수 없기 때문입니다.
> 또한 모든 분할에는 `random_state=seed`를 고정해야 동일한 데이터 분할 위에서 알고리즘 간 공정한 비교(Apples-to-Apples)와 **재현성(Reproducibility)**이 보장됩니다.

| 검증 기법                                | 사용 클래스/함수 (`sklearn.model_selection`)                                                  | 동작 방식                                                                                                                      | 장점 (Pros)                                                                                                                            | 단점 (Cons) 및 특징                                                                                                      |
| :--------------------------------------- | :-------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------- |
| **1. Train / Test Split**                | `train_test_split(X, Y, test_size=0.33, random_state=7)`                                      | 데이터를 학습용(예: 67%)과 테스트용(33%)으로 1회 분할                                                                          | 속도가 매우 빠름. 데이터가 매우 크거나(수백만 건) 모델 학습이 느릴 때 적합                                                             | 분할에 따라 성능 추정치의 **분산(Variance)이 큼**                                                                        |
| **2. k-fold Cross-Validation**           | `KFold(n_splits=10, random_state=7, shuffle=True)` + `cross_val_score(model, X, Y, cv=kfold)` | 데이터를 $k$개 폴드(보통 3, 5, 10)로 나누어 $k-1$개로 학습, 1개로 평가를 $k$번 반복 후 평균(`mean()`)과 표준편차(`std()`) 산출 | **머신러닝 모델 평가의 황금 표준(Gold Standard)**. 단일 분할보다 분산이 낮고 신뢰도가 높음 (`std()`는 **모델 안정성(Stability)** 지표) | 모델을 $k$번 학습해야 하므로 단일 분할보다 $k$배의 연산 시간 소요                                                        |
| **3. Leave One Out CV (LOOCV)**          | `LeaveOneOut()` + `cross_val_score`                                                           | 폴드 크기 $k=N$(전체 샘플 수)로 설정하여 매번 1개 샘플만 테스트셋으로 사용                                                     | 학습 데이터 활용 극대화                                                                                                                | 데이터 개수만큼 모델을 학습하므로 **연산 비용이 매우 크고**, 각 테스트 폴드가 1개뿐이라 결과의 표준편차(`std`)가 매우 큼 |
| **4. Repeated Random Test-Train Splits** | `ShuffleSplit(n_splits=10, test_size=0.33, random_state=7)`                                   | 무작위 Train/Test 분할을 지정한 횟수($n\_splits$)만큼 반복                                                                     | Train/Test 분할의 빠른 속도와 k-fold의 분산 감소 효과를 절충                                                                           | 반복 시 동일한 데이터가 여러 번 테스트/학습에 중복 포함되는 **데이터 중복(Redundancy)** 발생 가능                        |

---

## Lab 4: 분류 모델 평가 지표 및 ML 알고리즘 비교 (Compare ML Algorithms)

### 1. 분류 모델 5대 평가 지표 (Classification Evaluation Metrics)

> **Scikit-Learn `cross_val_score`의 부호 규칙 주의**: `cross_val_score`는 항상 **"점수가 클수록 좋다(오름차순 정렬 가능)"**는 통합 API 규칙을 따릅니다. 따라서 원래 작을수록 좋은 지표(예: LogLoss, MSE)는 앞에 `neg_`가 붙어 **음수 값으로 반환**됩니다.

| 평가 지표                                  | `scoring` 인자 또는 함수 (`sklearn.metrics`)                          | 개념 및 수식적 의미                                                                                                                                                                    | 불균형 데이터(Imbalanced Data)에서의 한계 및 활용법                                                                                                                                                                                                                                                         |
| :----------------------------------------- | :-------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1. Classification Accuracy (정확도)**    | `scoring='accuracy'`                                                  | 전체 예측 중 맞춘 비율. $\frac{TP+TN}{TP+TN+FP+FN}$                                                                                                                                    | **클래스 불균형이 심할 때 가장 오용되기 쉬운 지표**. 다수 클래스(0)로만 모두 예측해도 정확도가 93% 이상 나오므로 소수 클래스 예측력을 전혀 반영하지 못함                                                                                                                                                    |
| **2. Logarithmic Loss (로그 손실)**        | `scoring='neg_log_loss'`                                              | 클래스 소속 예측 확률 $\hat{y}_i$에 대해 정답과의 차이를 로그 우도(Log-Likelihood) 기반으로 처벌. $\text{LogLoss} = -\frac{1}{N}\sum [y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]$ | $0$에 가까울수록 완벽한 모델. 예측이 틀렸을 때 확신도(확률)가 높을수록 기하급수적으로 큰 페널티 부여                                                                                                                                                                                                        |
| **3. Area Under ROC Curve (AUROC / AUC)**  | `scoring='roc_auc'`, `roc_curve(y, probs)`, `roc_auc_score(y, probs)` | 모든 임계값(Cutpoint)에서의 **민감도(Sensitivity/Recall, $TPR = \frac{TP}{TP+FN}$)**와 **$1-\text{특이도}$($FPR = \frac{FP}{FP+TN}$)**의 트레이드오프를 그린 ROC 곡선 아래 면적        | $1.0$이면 완벽한 분류, $0.5$면 무작위 추측(가치 없음). $0.9\sim 1.0$(Excellent), $0.8\sim 0.9$(Good). **불균형 이진 분류에서 모델의 양성/음성 판별 능력(Discrimination)을 평가하는 가장 핵심적인 표준 지표**                                                                                                |
| **4. Confusion Matrix (혼동 행렬)**        | `confusion_matrix(Y_test, predicted)`                                 | 실제 클래스와 예측 클래스의 교차 표 (`[[TN, FP], [FN, TP]]`)                                                                                                                           | - **Type I Error (제1종 오류, False Positive)**: 실제 정상(0)인데 부도/질병(1)으로 잘못 예측<br>- **Type II Error (제2종 오류, False Negative)**: 실제 부도/질병(1)인데 정상(0)으로 놓침 → **신용평가(부도 예측) 및 의료 진단에서는 치명적인 손실을 유발하는 Type II Error(FN)를 줄이는 것이 훨씬 중요함!** |
| **5. Classification Report (분류 리포트)** | `classification_report(Y_test, predicted)`                            | 클래스별 **Precision(정밀도)**, **Recall(재현율)**, **F1-score(정밀도와 재현율의 조화평균)**, **Support(샘플 수)** 및 Macro/Weighted 평균 일괄 출력                                    | 불균형 데이터에서는 소수 클래스(1)의 Precision, Recall, F1-score와 Macro Avg를 중점적으로 확인해야 함                                                                                                                                                                                                       |

### 2. 알고리즘 스팟 체킹 (Spot-Checking) 및 공정한 비교

어떤 알고리즘이 해당 데이터셋에 가장 잘 맞을지 사전에 알 수 없으므로, **선형/비선형, 모수적(Parametric)/비모수적(Non-parametric), 인스턴스 기반/트리 기반** 등 다양한 표현형을 가진 알고리즘들을 동일한 테스트 환경(동일한 `random_state`의 10-fold CV)에서 실험해보고 박스플롯(`pyplot.boxplot(results)`)으로 평균과 분산(안정성)을 비교하여 후보군을 압축(Shortlisting)합니다.

| 알고리즘 약어 | Scikit-Learn 클래스                                            | 모델 특성 (모수/비모수, 선형/비선형)   | 핵심 원리 및 주의사항                                                                                                                                                                                                             |
| :------------ | :------------------------------------------------------------- | :------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **LR**        | `LogisticRegression` (`sklearn.linear_model`)                  | 모수적(Parametric) / 선형              | 입력 변수의 정규분포 가정, 선형 결정 경계, 높은 해석력                                                                                                                                                                            |
| **LDA**       | `LinearDiscriminantAnalysis` (`sklearn.discriminant_analysis`) | 모수적(Parametric) / 선형              | 클래스 간 분산은 최대화하고 클래스 내 분산은 최소화하는 선형 조합 탐색                                                                                                                                                            |
| **NB**        | `GaussianNB` (`sklearn.naive_bayes`)                           | 모수적(Parametric) / 확률 기반         | 베이즈 정리 및 조건부 독립 가정 기반 분류기                                                                                                                                                                                       |
| **KNN**       | `KNeighborsClassifier` (`sklearn.neighbors`)                   | 비모수적(Non-parametric) / 거리 기반   | 가장 가까운 $k$개 이웃의 다수결/평균으로 예측. 스케일에 민감하며 데이터가 크면 예측 속도가 느림                                                                                                                                   |
| **CART (DT)** | `DecisionTreeClassifier` (`sklearn.tree`)                      | 비모수적(Non-parametric) / 비선형 트리 | 지니 불순도(Gini index) 등의 비용 함수를 최소화하도록 탐욕적(Greedily)으로 이진 분할. `export_graphviz` / `plot_tree`로 시각화 가능. **과적합(Overfitting)되기 쉬우므로 `max_depth`, `min_samples_leaf`, 가지치기(Pruning) 필수** |
| **SVM**       | `SVC` (`sklearn.svm`)                                          | 비선형 커널 확장 가능                  | 클래스 간 마진(Margin)을 최대화하는 초평면 탐색 (기본적으로 RBF 커널 사용). 대용량 데이터에서는 학습 시간이 매우 오래 걸림                                                                                                        |

---

## Lab 5: 프로세스 자동화(Pipeline), 앙상블(Ensembles), 튜닝 및 모델 저장

### 1. 파이프라인 (Pipelining)을 통한 데이터 누수(Data Leakage) 방지

- **데이터 누수(Data Leakage)란?** 전체 데이터셋에 대해 먼저 스케일링(`StandardScaler`)이나 특성 추출(`PCA`, `SelectKBest`)을 수행한 뒤 교차검증을 돌리면, 테스트 폴드의 분포 정보(평균, 분산 등)가 학습 과정에 유출되어 성능이 과대평가되는 치명적 오류입니다.
- **해결책 (`sklearn.pipeline`)**:
  1. **`Pipeline(steps)`**: 전처리 단계와 모델 학습 단계를 하나의 체인(Estimator)으로 묶어, **교차검증의 각 폴드 내부에서 학습 폴드(`X_train`)로만 `fit`하고 테스트 폴드(`X_test`)에는 `transform`만 적용**되도록 강제합니다.
     ```python
     estimators = [('standardize', StandardScaler()), ('lda', LinearDiscriminantAnalysis())]
     model = Pipeline(estimators)
     ```
  2. **`FeatureUnion(transformer_list)`**: 여러 특성 추출/선택 기법(예: `PCA(n_components=3)` + `SelectKBest(k=6)`)을 병렬로 수행한 뒤 하나의 특성 행렬로 합쳐서 파이프라인의 단일 단계로 주입할 수 있습니다.

### 2. 앙상블 기법 (Ensemble Methods, `sklearn.ensemble`)

여러 개의 모델을 결합하여 단일 모델의 한계(높은 분산 또는 높은 편향)를 극복하는 기법입니다.

| 앙상블 유형                            | 핵심 원리                                                                                                                                                                                          | 대표 알고리즘 및 클래스                                                                                                                                                                                                                                                                  | 주요 하이퍼파라미터                                                      |
| :------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------- |
| **1. Bagging (Bootstrap Aggregation)** | 학습 데이터에서 **복원 추출(Bootstrap)**로 여러 서브샘플을 만들어 독립적으로 동일한 유형의 모델(주로 분산이 큰 무가지치기 결정 트리)을 병렬 학습한 뒤 예측을 평균/다수결 결합 (**분산 감소 효과**) | - **Bagged Decision Trees**: `BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=100)`<br>- **Random Forest**: `RandomForestClassifier(n_estimators=100, max_features=3)` (각 노드 분할 시 전체 변수가 아닌 무작위 부분집합만 고려하여 **트리 간 상관관계를 낮춤**) | `n_estimators` (트리 개수), `max_features` (분할 시 고려할 최대 특성 수) |
| **2. Boosting (부스팅)**               | 여러 모델을 **순차적으로(Sequentially)** 학습시키며, 이전 모델이 틀린(분류하기 어려운) 샘플에 더 높은 가중치를 부여하거나 잔차(오차)를 보정하도록 학습 (**편향 감소 효과**)                        | - **AdaBoost**: `AdaBoostClassifier(n_estimators=30, random_state=7)`<br>- _(과제/논문 확장)_: `GradientBoostingClassifier`, `XGBClassifier`                                                                                                                                             | `n_estimators`, `learning_rate`, `max_depth`                             |
| **3. Voting (보팅)**                   | 서로 **다른 유형(Heterogeneous)**의 독립된 모델(예: LR + CART + SVM)을 각각 학습시킨 후 예측 결과를 평균/다수결로 결합 (가중치 학습 시 Stacking으로 발전)                                          | - **Voting Classifier**: `VotingClassifier(estimators=[('logistic', model1), ('cart', model2), ('svm', model3)])`                                                                                                                                                                        | `estimators` (모델 리스트), `voting` (`'hard'` 또는 `'soft'`)            |

### 3. 하이퍼파라미터 튜닝 (Hyperparameter Tuning, `sklearn.model_selection`)

모델이 학습을 통해 스스로 찾는 가중치(Parameters)와 달리, 사용자가 사전에 지정해야 하는 설정값(Hyperparameters)을 최적화하는 과정입니다.

1. **Grid Search (`GridSearchCV(estimator, param_grid, cv=...)`)**:
   - 지정한 파라미터 그리드(`dict`)의 **모든 가능한 조합을 전수 조사**하여 최적 조합 도출.
   - 확인 속성: `grid.best_score_`, `grid.best_estimator_`, `grid.best_params_`.
2. **Random Search (`RandomizedSearchCV(estimator, param_distributions, n_iter=100, random_state=7)`)**:
   - 연속적인 확률분포(예: `scipy.stats.uniform()`)나 넓은 탐색 공간에서 지정한 횟수(`n_iter`)만큼 무작위로 조합을 추출하여 평가. 탐색 공간이 클 때 Grid Search보다 훨씬 효율적임.

### 4. 모델 직렬화 및 저장/로드 (Saving and Loading Models)

학습이 완료된 최종 모델을 디스크에 저장해 두었다가 추후 새로운 데이터 예측에 재사용합니다.

- **`pickle` 모듈**: 파이썬 표준 객체 직렬화 방식.
  - 저장: `pickle.dump(model, open('finalized_model.sav', 'wb'))`
  - 로드: `loaded_model = pickle.load(open('finalized_model.sav', 'rb'))`
- **`joblib` 라이브러리**: SciPy 생태계 도구로, **NumPy 배열이 많이 포함된 대용량 머신러닝 모델(예: Random Forest, KNN)을 훨씬 빠르고 효율적으로 저장/로드** (`import joblib` → `joblib.dump(model, filename)` / `joblib.load(filename)`).
- **모델 배포/재사용 시 주의사항**: 모델을 저장할 때와 로드할 때의 **Python 버전** 및 **주요 라이브러리(NumPy, Scikit-Learn 등) 버전**이 일치해야 역직렬화(Deserialization) 오류를 방지할 수 있습니다.

---

## Lab 6: Keras를 활용한 딥러닝 다층 퍼셉트론 (MLP with Keras)

### 1. Keras와 TensorFlow 백엔드

- **Keras (`tf.keras`)**: TensorFlow 2.0의 공식 고수준(High-level) API로, 직관적인 레이어 조립 방식으로 신경망을 빠르게 구축할 수 있으며 CPU와 다중 GPU 학습을 모두 지원합니다.
- 딥러닝은 가중치 초기화 및 확률적 경사하강법(SGD) 등 확률적 프로세스를 포함하므로 `numpy.random.seed(7)` (및 `tf.random.set_seed()`)로 난수 시드를 고정해야 재현성을 확보할 수 있습니다.

### 2. Keras 딥러닝 모델링 5단계 생명주기 (Lifecycle)

```python
# 1. 데이터 로드 및 X, Y 분리
X = dataset[:, 0:8]
Y = dataset[:, 8]

# 2. 모델 정의 (Define Model): Sequential 및 Dense 레이어 적층
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu')) # 은닉층 1 (입력 변수 8개 지정)
model.add(Dense(8, activation='relu'))               # 은닉층 2
model.add(Dense(1, activation='sigmoid'))            # 출력층 (이진 분류이므로 0~1 확률 출력)

# 3. 모델 컴파일 (Compile Model): 손실함수, 옵티마이저, 평가지표 설정
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 4. 모델 학습 (Fit Model): 에포크(반복 횟수) 및 배치 크기(가중치 업데이트 단위) 지정
model.fit(X, Y, epochs=150, batch_size=10)

# 5. 모델 평가 (Evaluate Model)
scores = model.evaluate(X, Y)
```

- **활성화 함수 (Activation Functions)**:
  - 은닉층에는 과거 `sigmoid`나 `tanh`보다 기울기 소실 문제가 적고 학습 성능이 뛰어난 **`relu` (Rectifier)**를 주로 사용합니다.
  - 이진 분류의 출력층에는 출력값을 $[0, 1]$ 사이의 확률로 매핑하는 **`sigmoid`**를 사용하며, 손실 함수로는 **`binary_crossentropy`**(로그 손실)를 짝지어 사용합니다.

### 3. 딥러닝 모델의 검증 데이터 분할 3가지 방식

학습 데이터로만 `evaluate`한 결과는 과적합 여부와 일반화 성능을 신뢰할 수 없으므로 반드시 검증 분할을 수행해야 합니다.

1. **자동 검증 분할 (`validation_split`)**: `model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10)`처럼 학습 데이터의 일정 비율(예: 뒤쪽 33%)을 자동으로 검증셋으로 분리하여 매 에포크마다 `val_loss`, `val_accuracy` 모니터링.
2. **수동 검증 분할 (`validation_data`)**: `train_test_split`으로 분리한 후 `model.fit(X_train, y_train, validation_data=(X_test, y_test), ...)`로 직접 주입.
3. **수동 k-Fold 교차검증 (`StratifiedKFold`)**: 클래스 비율을 유지하는 `StratifiedKFold(n_splits=10)`의 `for train, test in kfold.split(X, Y):` 루프 안에서 **매 폴드마다 `Sequential()` 모델을 새로 생성·컴파일·학습**시켜야 이전 폴드의 가중치가 이어지는 누수를 막을 수 있습니다.

### 4. Keras와 Scikit-Learn의 결합 (`KerasClassifier` / `scikeras`)

- Keras 모델을 생성·컴파일하여 반환하는 함수 `create_model(optimizer='rmsprop', init='glorot_uniform')`을 정의하고, 이를 래퍼 클래스(`KerasClassifier`)로 감싸면 Scikit-Learn의 `cross_val_score`와 `GridSearchCV`를 그대로 사용할 수 있습니다.
- **튜닝 가능한 주요 딥러닝 하이퍼파라미터**:
  - `optimizer`: `['rmsprop', 'adam']` (가중치 탐색 최적화 알고리즘)
  - `init` (`kernel_initializer`): `['glorot_uniform', 'normal', 'uniform']` (가중치 초기화 기법)
  - `epochs`: `[50, 100, 150]` (전체 데이터셋 반복 학습 횟수)
  - `batch_size`: `[5, 10, 20]` (가중치 1회 업데이트당 평가할 샘플 수)
- 주의: 딥러닝 Grid Search는 조합 수 $\times$ 폴드 수만큼 신경망을 처음부터 학습시키므로 연산량이 막대합니다. 따라서 먼저 **소규모 데이터 서브셋으로 실험을 설계하거나 GPU 가속을 활용**하는 것이 중요합니다.

---
