# 참고 논문 2편 (NOTE & NATE) 심층 요약 및 과제 적용 전략

본 문서는 과제 수행을 위해 필수적으로 참고해야 하는 `Paper/` 디렉토리의 논문 2편(**NOTE** 및 **NATE**)을 상세히 요약하고, 두 논문의 공통점과 차이점을 비교한 뒤, 이를 이번 **Give Me Some Credit (GMSC)** 과제에 어떻게 적용할지 구체적으로 정리한 문서입니다.

---

## 1. 두 논문의 핵심 비교 요약표

| 구분 | 논문 1: **NOTE** (2024.10) | 논문 2: **NATE** (2024.12) |
| :--- | :--- | :--- |
| **논문 제목** | *NOTE: non-parametric oversampling technique for explainable credit scoring* | *NATE: Non-pArameTric approach for Explainable credit scoring on imbalanced class* |
| **학술지** | Nature *Scientific Reports* 14, 26070 (2024) | *PLOS ONE* 19(12): e0316454 (2024) |
| **저자** | Seongil Han, Haemin Jung, Paul D. Yoo, Alessandro Provetti, Andrea Cali | Seongil Han, Haemin Jung |
| **파일명** | `Paper/s41598-024-78055-5.pdf` | `Paper/journal.pone.0316454.pdf` |
| **공통 문제의식** | 1. 신용평가 데이터의 **심각한 클래스 불균형(Imbalanced Class)** (정상 고객 다수 vs 부도 고객 소수)<br>2. 데이터의 **비선형성(Non-linearity)**으로 인한 전통적 선형 모델(Logistic Regression)의 예측 한계<br>3. 고성능 비모수(Non-parametric) 트리 모델의 **블랙박스 한계와 설명 가능성(Explainability, XAI) 부재** | 좌동 |
| **사용 데이터셋** | 1. Home Equity (HE, 5,960건, IR=4.012)<br>2. **Give Me Some Credit (GMSC, 150,000건, 8개 파생/전처리 특성, IR=13.961)** | **Give Me Some Credit (GMSC, 150,000건, IR=13.961)** |
| **특성 추출 (Feature Extraction)** | **NSA (Non-parametric Stacked Autoencoder)**로 비선형 잠재 변수(Latent vectors, 2개: `en1`, `en2`) 추출 후 원본 특성과 결합(Concatenation) (비교군: rSVD) | 별도 딥러닝 오토인코더 없이 전처리 및 파생 특성(`CombinedDefaulted`, `CombinedCreditLoans`) 활용 |
| **불균형 해소 (Resampling)** | **cWGAN-GP (Conditional Wasserstein GAN with Gradient Penalty)** 기반 딥러닝 조건부 생성 오버샘플링 ($1:1$ 완전 균형, 총 279,948건) | **SMOTE (오버샘플링)** vs **NearMiss (언더샘플링)**을 다양한 불균형 비율(IR = $13.96 \rightarrow 6.67 \rightarrow 3.12 \rightarrow 2 \rightarrow 1$)로 다각도 비교 |
| **비교 분류기 (Classifiers)** | LR, DT, **ET (Extra Trees)**, **RF (Random Forest)**, **GB (Gradient Boosting)** | LR, LDA, KNN, DT, NB, **RF**, **GB**, **XGB (XGBoost)** |
| **평가지표 및 검증** | 5-fold CV Grid Search 튜닝 후 **10-fold CV** 평가, 주 지표: **AUROC**, 통계 검정: **Friedman Test** | **5-fold CV** Grid Search 튜닝 및 평가, 주 지표: **AUROC, MCC, F1-Score** (Accuracy의 한계 지적) |
| **GMSC 최고 성능** | **NOTE + Random Forest: AUROC = 0.9837** (GB: 0.9834, ET: 0.9831, LR: 0.9750, DT: 0.9648) | **SMOTE (IR=1) + Gradient Boosting: AUROC = 0.9649, MCC = 0.8104, F1 = 0.9072** (NearMiss + XGB: AUROC = 0.9451, 단 35.27초 소요) |
| **설명 가능성 (XAI)** | **SHAP (`TreeExplainer`, `LinearExplainer`)**를 통한 전역(Global) 및 국소(Local) 특성 기여도 분석 | **SHAP (`TreeExplainer`, `LinearExplainer`)**의 Force Plot, Decision Plot, Bar Plot, Violin Plot 분석 |

---

## 2. 논문 1 상세 요약: NOTE (Nature Scientific Reports, 2024)

### (1) 연구 배경 및 기존 기법의 한계
- 금융권 신용평가(Credit Scoring)의 표준 모델인 **로지스틱 회귀(LR)**는 모수적(Parametric) 선형 모델로서 해석이 쉽고 연산이 빠르지만, 변수 간 비선형 관계가 강한 데이터에서는 예측력이 떨어집니다.
- 반면 **Random Forest(RF), Extra Trees(ET), Gradient Boosting(GB)** 같은 비모수적(Non-parametric) 트리 기반 앙상블 모델은 비선형 데이터를 잘 학습하지만 예측 과정을 설명하기 어렵고, 클래스 불균형(GMSC의 경우 부도 클래스가 `6.684%`, 불균형 비율 $\text{IR} = 13.961$)이 심할 때 다수 클래스(정상)로 편향됩니다.
- 기존의 대표적인 오버샘플링 기법인 **SMOTE**는 기존 소수 클래스 샘플 간 단순 선형 보간에 의존하므로 고차원·비선형 데이터에서 **클래스 중첩(Class overlap)과 노이즈(Noise)를 유발하여 과적합(Overfitting)**을 초래할 수 있습니다.
- 또한 선형 차원 축소 기법인 **PCA**나 **rSVD(Randomized SVD)**는 비선형 잠재 패턴을 온전히 포착하지 못합니다.

### (2) NOTE 프레임워크의 4단계 구조 (Sequential Stages)
```
[Stage 1] 신용평가 데이터셋 수집 및 전처리 (GMSC: 150,000건, 정상 139,974 / 부도 10,026)
   ↓
[Stage 2] NSA (Non-parametric Stacked Autoencoder) 기반 비선형 잠재 특성(Latent Features) 추출 및 원본 데이터와 병합 (Concatenation)
   ↓
[Stage 3] cWGAN-GP (Conditional Wasserstein GAN with Gradient Penalty)를 이용해 학습 데이터의 소수 클래스(부도=1)를 다수 클래스 수만큼 오버샘플링 (10,026건 → +129,948건 생성 = 139,974건)
   ↓
[Stage 4] 트리 기반 비모수 분류기(DT, ET, RF, GB) 및 LR 학습 (GridSearch 튜닝 + 트리 가지치기) 후 SHAP (TreeExplainer / LinearExplainer)으로 결과 해석
```
> **중요 아키텍처 포인트 (논문 Fig. 5)**: NSA를 통한 잠재 특성 병합 후, 데이터를 **Train(70%) / Test(30%)**로 나눈 뒤 **오직 Train 세트의 소수 클래스(Minority class)에 대해서만 cWGAN 오버샘플링을 수행**하여 Augmented Minority Class를 만들고, Test 세트는 원본 분포 그대로 평가합니다.

### (3) 핵심 구성 기술 상세
1. **NSA (Non-parametric Stacked Autoencoder)를 통한 특성 생성**:
   - 인코더(Encoder)와 디코더(Decoder)로 구성된 비지도 신경망으로, 입력 벡터 $v$와 복원 벡터 $v'$ 사이의 평균제곱오차 $L(v, v') = \frac{1}{n}\sum \|v - v'\|^2$를 최소화하도록 학습합니다.
   - 병목 구간(Coding layer $h$)에서 추출된 비선형 잠재 표현(Latent representations, 예: `en1`, `en2`)을 **원본 특성에 대체하는 것이 아니라 옆으로 이어 붙여(Concatenation)** 특성 공간을 확장함으로써 분류기가 비선형 구조를 더 잘 학습하도록 돕습니다.
2. **cWGAN-GP 기반 정형 데이터 오버샘플링**:
   - 일반 GAN이나 cGAN은 쿨백-라이블러 발산(KLD) / 젠센-샤논 발산(JSD)을 최소화하는 과정에서 실제 분포와 생성 분포의 지지집합(Support)이 겹치지 않으면 **모드 붕괴(Mode Collapse)**와 **기울기 소실/학습 불안정(Training Instability)**이 발생합니다.
   - 이를 해결하기 위해 최적 운송 비용(Optimal Transport Cost) 개념인 **Wasserstein 거리(Earth Mover's Distance)**를 손실 함수로 도입하고, 판별자(Discriminator)가 1-Lipschitz 연속성 조건을 만족하도록 가중치 클리핑 대신 **기울기 페널티(Gradient Penalty, WGAN-GP)**를 적용했습니다.
   - 나아가 생성자($G$)와 판별자($D$) 모두에 클래스 레이블 조건 $y$와 **보조 분류기 손실(Auxiliary Classifier, AC loss)**을 부여하고, 범주형 변수는 **Gumbel-Softmax** 및 **임베딩 레이어(Embedding layers)**로, 변수 간 고차 상호작용은 **병렬 교차 레이어(Parallel Crosslayers)**로 처리하여 정형 데이터(연속형+범주형 혼합) 생성 품질을 극대화했습니다.
3. **하이퍼파라미터 튜닝 및 과적합 방지 (Pruning)**:
   - 오버샘플링된 데이터에서 트리 모델이 과적합되는 것을 막기 위해 `max_depth=[1, 20]`, `min_samples_split={1, 2, 5, 10}`, `min_samples_leaf={1, 2, 4, 8}` 등을 5-fold CV Grid Search로 탐색하여 가지치기(Pruning) 효과를 주었습니다.

### (4) GMSC 데이터셋 실험 결과 및 통계적 검증
- **오버샘플링 기법별 AUROC 비교 (Table 3, GMSC 기준)**:
  - **Random Forest (RF)**: 원본(None) `0.8181` $\rightarrow$ DeepSMOTE `0.8530` $\rightarrow$ ADS-GAN `0.9766` $\rightarrow$ **NOTE `0.9837` (최고 성능)**
  - **Gradient Boosting (GB)**: 원본 `0.8153` $\rightarrow$ **NOTE `0.9834`**
  - **Extra Trees (ET)**: 원본 `0.7926` $\rightarrow$ **NOTE `0.9831`**
  - **Logistic Regression (LR)**: 원본 `0.7984` $\rightarrow$ **NOTE `0.9750`**
  - **Decision Tree (DT)**: 원본 `0.5814` $\rightarrow$ **NOTE `0.9648`**
- **특성 추출 기법별 비교 (Table 4)**: 선형 차원 축소/노이즈 제거 기법인 **rSVD** 단독(`RF: 0.9806`)이나 **rSVD+NSA**(`RF: 0.9816`)보다 **NSA 단독 적용(NOTE, `RF: 0.9837`)**이 모든 분류기에서 가장 높은 AUROC를 기록했습니다.
- **Friedman 비모수 통계 검정 (Table 5)**: 다중 모델에 대한 반복 측정 순위 검정인 Friedman test 결과 $p\text{-value} = 0.0224 < 0.05$ (오버샘플링 비교) 및 $p\text{-value} = 0.0194 < 0.05$ (특성 추출 비교)로 NOTE의 성능 우위가 통계적으로 유의미함을 입증했습니다.

---

## 3. 논문 2 상세 요약: NATE (PLOS ONE, 2024)

### (1) 연구 배경 및 차별점
- NOTE 논문이 무거운 딥러닝 생성 모델(NSA + cWGAN-GP)에 초점을 맞췄다면, **NATE** 논문은 실제 머신러닝 실습 및 실무에서 가장 널리 쓰이는 전통적 리샘플링 기법인 **SMOTE(오버샘플링)**와 **NearMiss(언더샘플링)**를 활용하여, **불균형 비율(Imbalance Ratio, IR)의 단계적 변화가 모수적 모델(LR)과 비모수적 트리 앙상블 모델(RF, GB, XGB)의 성능 및 연산 효율성에 미치는 영향**을 심층 분석했습니다.

### (2) NATE 프레임워크의 리샘플링 설계 (IR 단계별 실험)
원본 GMSC 데이터셋(정상 `139,974`건, 부도 `10,026`건, 부도 비율 `7%`, $\text{IR} = 13.96$)으로부터 부도 클래스 비중을 **13% ($\text{IR}=6.67$), 24% ($\text{IR}=3.12$), 33% ($\text{IR}=2$), 50% ($\text{IR}=1$, 완전 균형)**로 조절하여 총 9개의 데이터셋 구성(원본 1 + NearMiss 4 + SMOTE 4)을 비교했습니다.
1. **NearMiss (언더샘플링)**: 거리(KNN) 기반으로 소수 클래스 샘플과 가장 가까운 다수 클래스(정상) 샘플만 남기고 나머지를 제거하여 $\text{IR}=1$일 때 정상 `10,026`건 : 부도 `10,026`건(총 `20,052`건)으로 축소.
2. **SMOTE (오버샘플링)**: 소수 클래스 샘플 $x_i$와 KNN으로 찾은 이웃 소수 클래스 샘플 $\hat{x}_i$ 사이를 무작위 비율 $\lambda \in [0, 1]$로 선형 보간($x_{\text{synthetic}} = x_i + \lambda(\hat{x}_i - x_i)$)하여 $\text{IR}=1$일 때 정상 `139,974`건 : 부도 `139,974`건(총 `279,948`건)으로 증강.

### (3) 평가지표 선정의 핵심 논리 (왜 Accuracy를 쓰면 안 되는가?)
- **Table 5 (원본 불균형 데이터 5-fold CV 결과)**에서 볼 수 있듯, 원본 데이터에서는 거의 모든 분류기(LR `0.9323`, LDA `0.9325`, KNN `0.9310`, RF `0.9322`, GB `0.9343`, XGB `0.9335`)가 **Accuracy는 93% 내외로 거의 똑같아 보이지만, 실제 F1-score는 LR이 `0.0539`, KNN이 `0.0525`에 불과**합니다.
- 즉, 다수 클래스(정상, 93.3%)에 편향되어 부도 고객을 전혀 잡아내지 못해도 Accuracy는 높게 나오는 착시가 발생하므로, 불균형 신용평가에서는 반드시 다음 3가지 지표를 함께 봐야 합니다:
  1. **AUROC (Area Under ROC Curve)**: 임계값에 무관한 전체적인 클래스 판별력 표준 지표.
  2. **MCC (Matthews Correlation Coefficient)**: $TP, TN, FP, FN$ 4가지 혼동 행렬 요소를 모두 고려하여 클래스 불균형에 가장 강건한 상관 계수($-1 \sim +1$).
  3. **F1-Score**: 정밀도(Precision)와 재현율(Recall)의 조화평균.

### (4) 주요 실험 결과 및 연산 시간 트레이드오프 분석
1. **불균형 비율(IR) 감소에 따른 성능 향상 (Table 7)**:
   - 비모수 트리 앙상블 모델(RF, GB, XGB)은 SMOTE와 NearMiss 모두에서 **IR이 낮아질수록(데이터가 $50:50$, $\text{IR}=1$에 가까워질수록) AUROC, MCC, F1이 일관되게 대폭 상승**했습니다.
   - 반면 **로지스틱 회귀(LR)**는 SMOTE 적용 시 MCC와 F1은 오르지만 AUROC는 원본(`0.7716`) 대비 $\text{IR}=1$에서 `0.7591`로 오히려 약 1%p 하락했습니다. 이는 선형 경계를 갖는 LR이 비선형 데이터의 오버샘플링 이점을 충분히 흡수하지 못함을 보여줍니다.
2. **SMOTE (오버샘플링) vs NearMiss (언더샘플링) 비교 (Table 8, 9)**:
   - **최고 예측 성능**: **SMOTE ($\text{IR}=1$) + Gradient Boosting (GB)** 조합이 **AUROC `0.9649`, MCC `0.8104`, F1 `0.9072`**로 전체 최고 성능을 달성했습니다 (동일 조건 XGB는 AUROC `0.9611`, RF는 `0.9311`). 이는 정보 손실 없이 전체 데이터를 보존하는 오버샘플링이 트리 앙상블 모델과 결합될 때 가장 강력함을 뜻합니다.
   - **연산 효율성 대 예측력의 트레이드오프 (Table 8)**:
     - **GB + SMOTE ($\text{IR}=1$, 27.9만 건)**: AUROC `0.9649`를 달성했으나 학습 및 평가에 **약 2,257초(약 37.6분)**가 소요됨.
     - **XGB + NearMiss ($\text{IR}=1$, 2만 건)**: 정보 손실로 인해 AUROC는 `0.9451`(MCC `0.7506`, F1 `0.8728`)로 소폭 낮지만, 소요 시간은 **단 35.27초(약 64배 빠름!)**에 불과함.
     - **실무적 시사점**: 정확도가 최우선인 심사에서는 **SMOTE + GB/XGB/RF**가 적합하고, 컴퓨팅 자원이 제한적이거나 실시간(Real-time) 빠른 재학습이 필요한 환경에서는 **NearMiss + XGBoost**가 매우 훌륭한 대안이 됩니다.

### (5) SHAP (`TreeExplainer`)을 통한 설명 가능성(XAI) 분석 결과 (두 논문 공통)
협조적 게임 이론의 **Shapley Value**에 기반한 **SHAP**($g(z') = \phi_0 + \sum_{i=1}^M \phi_i z'_i$)은 **국소적 정확성(Local Accuracy), 결측성(Missingness), 일관성(Consistency)**의 3대 공리를 만족하며 블랙박스 트리 모델을 투명하게 설명합니다.
- **전역적 해석 (Global Interpretability - Bar Plot & Violin Plot)**:
  - GMSC 데이터셋 전체에서 부도 예측에 가장 큰 영향을 미치는 상위 3대 핵심 변수는 다음과 같이 밝혀졌습니다:
    1. **`CombinedDefaulted`** (평균 $|\text{SHAP}| \approx 0.31$, 압도적 1위): 과거 30~89일 또는 90일 이상 연체 이력 유무.
    2. **`RevolvingUtilizationOfUnsecuredLines`** (평균 $|\text{SHAP}| \approx 0.22$, 2위): 신용카드 및 마이너스 통장 한도 소진율.
    3. **`NumberOfTime30-59DaysPastDueNotWorse`** (평균 $|\text{SHAP}| \approx 0.21$, 3위): 최근 2년간 단기(30~59일) 연체 횟수.
- **국소적 해석 (Local Interpretability - Force Plot & Decision Plot)**:
  - 개별 대출 신청자 한 명에 대해 **왜 이 고객의 부도 확률이 0.22(정상 판정) 또는 0.76(부도 판정)으로 산출되었는지** 각 변수가 확률을 높였는지(빨간색/양의 기여) 낮췄는지(파란색/음의 기여) 정량적으로 제시할 수 있습니다.
  - 예: 부도 확률이 `0.76`으로 예측된 고객의 경우, `CombinedDefaulted = 1`($+0.18$)과 높은 `RevolvingUtilizationOfUnsecuredLines = 0.796`($+0.09$)이 부도 위험을 크게 끌어올린 주원인임을 금융 실무자와 고객에게 명확히 설명(Evaluative AI)할 수 있습니다.

---

## 4. 우리 과제(`Data/givemesomecredit.csv`)와의 직접적인 연결 고리

과제에 제공된 `Data/givemesomecredit.csv` 파일의 첫 행(헤더)을 확인해 보면 다음과 같이 **총 9개 컬럼(타겟 1개 + 입력 변수 8개, 총 150,000행)**으로 구성되어 있습니다:
1. `SeriousDlqin2yrs` (Target $Y$: `0` = 139,974건, `1` = 10,026건)
2. `RevolvingUtilizationOfUnsecuredLines`
3. `age`
4. `NumberOfTime30-59DaysPastDueNotWorse`
5. `DebtRatio`
6. `MonthlyIncome`
7. `NumberOfDependents`
8. `CombinedDefaulted`
9. `CombinedCreditLoans`

> **핵심 발견!**
> 원래 Kaggle의 Give Me Some Credit 원본 데이터는 10개의 입력 변수를 가지고 있었고 `MonthlyIncome`과 `NumberOfDependents`에 결측치가 존재했습니다(NATE 논문의 Table 2 참고).
> 하지만 **우리 과제에 제공된 `Data/givemesomecredit.csv`는 NOTE 논문의 Table 1 및 두 논문의 SHAP 분석(Fig. 8)에 등장하는 바로 그 전처리·파생변수 생성이 완료된 8개 특성 버전**입니다!
> - 즉, 원본의 `NumberOfOpenCreditLinesAndLoans`와 `NumberRealEstateLoansOrLines`가 **`CombinedCreditLoans`**라는 범주형(0/1) 변수로 통합되었고,
> - 원본의 중장기 연체 변수(`NumberOfTime60-89DaysPastDueNotWorse`, `NumberOfTimes90DaysLate`)가 **`CombinedDefaulted`**라는 범주형(0/1) 변수로 통합되었으며,
> - 결측치(`MonthlyIncome`의 중앙값 대체 등)가 이미 깔끔하게 처리되어 있습니다.

### 과제 수행 시 두 논문을 벤치마크로 활용하는 핵심 전략
1. **스토리라인 구축**:
   - **Baseline (Lab 1~4 적용)**: 원본 불균형 데이터(`IR = 13.96`)에서 Logistic Regression 등 기본 모델을 돌렸을 때 **Accuracy(약 93.2%)의 함정**과 **낮은 F1-score(약 0.05) 및 AUROC(약 0.77~0.82)**를 먼저 보여줍니다.
   - **NATE 접근법 적용 (전통적 리샘플링 + 트리 앙상블)**: `SMOTE` 오버샘플링과 `NearMiss` 언더샘플링을 적용하고, 모수적 모델(`LR`, `LDA`, `NB`) 대비 비모수적 트리 앙상블 모델(`RandomForest`, `ExtraTrees`, `GradientBoosting`, `XGBoost`)의 성능 향상 및 **정확도 vs 학습 시간 트레이드오프**를 비교합니다 (목표 벤치마크: NATE의 GB+SMOTE **AUROC 0.9649**).
   - **NOTE 접근법 아이디어 접목 (비선형 잠재 특성 추출 + 딥러닝/고급 오버샘플링)**: Lab 3의 선형 차원 축소(`PCA`)와 Lab 6의 Keras 신경망(`Sequential`, `Dense`) 기술을 응용해 간단한 **Stacked Autoencoder(NSA)**로 비선형 잠재 변수(`en1`, `en2`)를 추출해 원본 8개 변수에 결합(총 10개 변수)하거나, 리샘플링과 결합하여 성능을 한 단계 더 끌어올리는 실험을 제시합니다 (목표 벤치마크: NOTE의 RF **AUROC 0.9837**).
   - **XAI (설명 가능한 AI) 마무리**: 최고 성능 모델에 대해 **트리 특성 중요도(Lab 3)**뿐만 아니라 두 논문의 핵심인 **SHAP (`TreeExplainer`)**을 적용하여 `CombinedDefaulted`, `RevolvingUtilizationOfUnsecuredLines` 등의 전역적 중요도와 개별 고객의 부도 예측 근거를 시각화함으로써 **예측력(Predictive Power)과 설명 가능성(Explainability)을 동시에 달성**하는 완벽한 결론을 도출할 수 있습니다.

