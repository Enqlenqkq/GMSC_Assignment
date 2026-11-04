# 더 빠른 학습을 위한 GPU 가속 완벽 가이드

`goal.txt`의 4번 질문(**"더 빠른 학습을 위해 제 GPU를 이용할 수 있나요? 하려면 방법은 어떻게 될까요?"**)에 대한 상세 답변 및 환경 구축 가이드입니다.

---

## 1. 핵심 결론 요약

**네, 본인의 GPU(특히 NVIDIA GeForce / RTX 시리즈)를 활용하여 학습 및 하이퍼파라미터 튜닝 시간을 수십 배 단축할 수 있습니다!**

특히 이번 과제 데이터셋(`Data/givemesomecredit.csv`)은 **150,000건**(오버샘플링 시 **약 280,000건**)으로 크기가 꽤 크며, 실제로 NATE 논문(Table 8)에서도 32코어 고성능 서버 CPU로 `SMOTE + Gradient Boosting`을 튜닝·평가하는 데 **약 2,257초(약 38분)**가 걸렸습니다.

단, **"어떤 라이브러리와 알고리즘을 쓰느냐"**에 따라 GPU 지원 여부와 활성화 방법이 완전히 다르므로 아래 표를 먼저 확인해야 합니다.

| 알고리즘 / 작업 영역 | 기본 라이브러리 (CPU) | GPU 가속 지원 여부 및 전환 방법 | 체감 속도 향상 |
| :--- | :--- | :--- | :--- |
| **Gradient Boosting / XGBoost** | `sklearn.ensemble.GradientBoostingClassifier` (단일 코어 CPU, **매우 느림**) | **`xgboost.XGBClassifier(tree_method='hist', device='cuda')`** 또는 `CatBoostClassifier(task_type='GPU')` | **약 20배 ~ 50배 이상 단축** |
| **Random Forest** | `sklearn.ensemble.RandomForestClassifier` (CPU 멀티코어 `n_jobs=-1`만 지원) | 1) **`xgboost.XGBRFClassifier(tree_method='hist', device='cuda')`**<br>2) NVIDIA RAPIDS **`cuml.ensemble.RandomForestClassifier`**<br>3) CPU 사용 시 반드시 **`n_jobs=-1`** 지정 | `n_jobs=-1`로 CPU 코어 수만큼 단축 / GPU 전환 시 **10~30배 단축** |
| **MLP / Autoencoder(NSA) / cWGAN (Lab 6, 7 & NOTE)** | `tensorflow` / `keras` (CPU 모드) | **CUDA GPU가 활성화된 TensorFlow / PyTorch 백엔드** 사용 (자동으로 GPU 텐서 연산 수행) | **약 10배 ~ 40배 이상 단축** |
| **GridSearchCV / 교차검증 (Lab 5)** | `GridSearchCV(..., n_jobs=1)` | CPU 모델은 **`n_jobs=-1`** 필수 / GPU 모델(`XGBClassifier` 등)은 모델 내부에서 GPU를 쓰므로 `GridSearchCV`의 `n_jobs=1` 또는 `2`로 설정 (GPU 메모리 충돌 방지) | 탐색 시간 대폭 단축 |
| **SHAP 값 계산 (XAI)** | `shap.TreeExplainer(model)` (CPU) | XGBoost GPU 내장 기여도 계산 또는 **`shap.GPUTreeExplainer`** | 대용량 샘플 해석 시간 **10배 이상 단축** |

---

## 2. 내 PC의 GPU 확인 및 준비 단계 (Windows 기준)

먼저 본인의 PC에 **NVIDIA GPU**가 장착되어 있는지, 그리고 드라이버(CUDA)가 설치되어 있는지 확인합니다.
- **확인 방법**: Windows 작업 관리자(`Ctrl + Shift + Esc`) $\rightarrow$ `[성능]` 탭 하단에서 **GPU 0 / GPU 1**의 모델명(예: `NVIDIA GeForce RTX 3060 / 4060 / 4070` 등)을 확인하거나, 터미널(PowerShell/CMD)에서 다음 명령어를 실행합니다:
  ```powershell
  nvidia-smi
  ```
  - 이 명령어가 정상 실행되면서 우측 상단에 `CUDA Version: 12.x` (또는 `11.x`)와 GPU 이름이 출력된다면 GPU 가속을 사용할 준비가 완료된 것입니다.
  - 만약 AMD 라데온(Radeon)이나 Intel 내장 그래픽이라면 CUDA 기반 라이브러리는 로컬에서 직접 가속하기 어려우므로, 아래 **4장의 Google Colab 무료 GPU 활용법**이나 **`n_jobs=-1` CPU 병렬화**를 사용하는 것이 좋습니다.

---

## 3. 모델 및 작업별 구체적인 GPU 활용 방법

### 방법 1: 트리 부스팅 & 랜덤 포레스트를 GPU로 초고속 학습하기 (가장 추천!)

Scikit-Learn의 기본 `GradientBoostingClassifier`는 트리를 하나씩 순차적으로 만들면서 CPU 단일 스레드만 사용하기 때문에 28만 건 데이터에서 극도로 느립니다. 이를 **XGBoost의 GPU 모드**로 바꾸거나 파라미터만 설정해도 속도가 비약적으로 빨라집니다.

#### ① XGBoost (`XGBClassifier`, `XGBRFClassifier`) GPU 설정
설치: `pip install xgboost` (별도의 복잡한 CUDA 툴킷 설치 없이도 최신 `pip` 패키지에 GPU 커널이 기본 포함되어 있어 NVIDIA 드라이버만 있으면 바로 작동합니다.)

```python
from xgboost import XGBClassifier, XGBRFClassifier

# 1. GPU 가속 Gradient Boosting (XGBoost 2.0 이상 표준 문법)
xgb_gpu = XGBClassifier(
    tree_method="hist",     # 히스토그램 기반 고속 트리 분할 알고리즘
    device="cuda",          # GPU(CUDA) 가속 활성화! (구버전은 tree_method='gpu_hist')
    n_estimators=500,
    max_depth=6,
    learning_rate=0.1,
    random_state=7,
    eval_metric="auc"
)

# 2. GPU 가속 Random Forest (XGBoost 내장 Random Forest API)
rf_gpu = XGBRFClassifier(
    tree_method="hist",
    device="cuda",          # Random Forest도 GPU에서 초고속 학습 가능!
    n_estimators=500,
    max_depth=10,
    random_state=7
)
```

> **주의사항**: `device="cuda"`를 켠 모델을 `GridSearchCV`나 `cross_val_score`에 넣을 때는, 이미 GPU 1개가 모든 연산을 병렬 처리하므로 `GridSearchCV(..., n_jobs=1)`로 두는 것이 좋습니다. `n_jobs=-1`을 동시에 주면 여러 프로세스가 한꺼번에 GPU 메모리를 점유하려다 속도 저하나 메모리 부족(`Out of Memory`)이 발생할 수 있습니다.

#### ② Scikit-Learn 기본 모델 사용 시 필수 고속화 설정 (`n_jobs=-1` & `HistGradientBoosting`)
과제 요건상 순수 `scikit-learn` 클래스만 써야 하는 구간에서는 다음 2가지를 반드시 적용하세요:
1. **멀티코어 병렬화 (`n_jobs=-1`)**:
   - `RandomForestClassifier(n_estimators=100, n_jobs=-1)`
   - `ExtraTreesClassifier(n_estimators=100, n_jobs=-1)`
   - `KNeighborsClassifier(n_jobs=-1)`
   - `GridSearchCV(..., n_jobs=-1)` / `RandomizedSearchCV(..., n_jobs=-1)`
   - 내 CPU가 8코어 16스레드라면 `n_jobs=-1` 옵션 하나만 추가해도 학습 속도가 **약 6~10배** 빨라집니다.
2. **`GradientBoostingClassifier` 대신 `HistGradientBoostingClassifier` 활용**:
   - Scikit-Learn이 LightGBM의 히스토그램 기법과 멀티스레딩을 도입해 공식 제공하는 **`from sklearn.ensemble import HistGradientBoostingClassifier`**를 사용하면, 동일한 Scikit-Learn 환경에서도 기존 `GradientBoostingClassifier`보다 **10배 이상 빠르게** 학습됩니다.

---

### 방법 2: 딥러닝 (Lab 6 MLP, Lab 7 CNN, NOTE 논문의 NSA 오토인코더 & cWGAN) GPU 활용하기

딥러닝 행렬 연산은 GPU에서 가장 극적인 가속 효과를 볼 수 있습니다. 그런데 **Windows 환경에서는 매우 중요한 버전 규칙**이 하나 있습니다.

> **[중요] Windows에서 TensorFlow GPU 지원 변경사항**:
> Google은 **TensorFlow 2.11 버전부터 Windows Native 환경에서의 공식 GPU 지원(`tensorflow-gpu`)을 중단**했습니다 (Windows Native에서는 TF 2.10까지만 GPU 지원).
> 따라서 Windows 로컬 PC에서 Keras(`Sequential`, `Dense` 등)를 GPU로 돌리려면 아래 **3가지 해결책 중 하나**를 선택하면 됩니다.

#### 해결책 A (가장 추천 & 최신 방식): **Keras 3 + PyTorch GPU 백엔드** 사용
최신 **Keras 3 (`keras >= 3.0`)**는 코드 한 줄 바꾸지 않고도 내부 백엔드 엔진만 TensorFlow 대신 **PyTorch(Windows GPU 완벽 지원!)**로 지정하여 실행할 수 있습니다.
1. PyTorch GPU 버전 및 Keras 설치:
   ```powershell
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   pip install keras
   ```
2. 파이썬 코드 최상단에 백엔드 지정 후 Lab 6 / Lab 7과 100% 동일한 코드 사용:
   ```python
   import os
   os.environ["KERAS_BACKEND"] = "torch"  # Keras가 Windows GPU를 지원하는 PyTorch 엔진을 사용하도록 설정!

   import torch
   import keras
   from keras.models import Sequential
   from keras.layers import Dense

   print("GPU 사용 가능 여부:", torch.cuda.is_available())
   print("장착된 GPU 이름:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
   ```
   이렇게 하면 Lab 6의 MLP, Lab 7의 CNN, 그리고 NOTE 논문의 Stacked Autoencoder(NSA)까지 **코드를 전혀 바꾸지 않고 내 Windows PC의 최신 RTX GPU에서 초고속으로 학습**시킬 수 있습니다!

#### 해결책 B: Windows WSL2 (Ubuntu)에서 TensorFlow GPU 실행
Windows 10/11에 내장된 WSL2 리눅스 환경을 사용하는 경우:
```bash
pip install "tensorflow[and-cuda]"
```
명령어 한 줄로 최신 TensorFlow와 CUDA/cuDNN이 함께 설치되어 `tf.config.list_physical_devices('GPU')`로 즉시 GPU가 인식됩니다. (WSL2에서는 Scikit-Learn과 100% 호환되는 NVIDIA의 GPU 머신러닝 라이브러리인 **RAPIDS `cuML`**도 설치 가능하여 Random Forest, SVM, KNN, PCA까지 전부 GPU로 돌릴 수 있습니다.)

#### 해결책 C: Windows Native에서 TensorFlow 2.10 고정 사용
Python 3.9 또는 3.10 가상환경(Conda 등)을 만들고 `conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0` 후 `pip install "tensorflow<2.11"`을 설치하면 기존 TensorFlow 2.10으로 Windows에서 GPU를 직접 인식시킬 수 있습니다.

---

### 방법 3: 딥러닝/MLP 학습 시 배치 크기(`batch_size`) 조절을 통한 GPU 효율 극대화

Lab 6 예제에서는 데이터셋(`pima-indians-diabetes.csv`)이 768건밖에 되지 않아 `batch_size=10`이나 `batch_size=5`를 사용했습니다.
하지만 **우리 과제의 GMSC 데이터셋은 150,000건(SMOTE 적용 시 280,000건)**입니다!
- 만약 28만 건 데이터에서 `batch_size=10`으로 150 에포크를 돌리면, 1 에포크당 무려 **28,000번의 가중치 업데이트**가 일어나므로 GPU를 써도 병목(오버헤드) 때문에 매우 느려집니다.
- **GPU를 사용할 때는 `batch_size`를 `256`, `512`, 또는 `1024`로 큼직하게 늘려주어야** GPU의 수천 개 CUDA 코어가 한 번에 데이터를 병렬 처리하여 학습 속도가 수십 배 빨라집니다!
- 또한 `keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)` 콜백을 함께 사용하면 불필요한 에포크 반복을 자동으로 멈추어 시간도 아끼고 과적합도 방지할 수 있습니다.

---

## 4. 로컬 GPU 설정이 번거로울 때의 대안: Google Colab GPU 런타임

Lab 1과 Lab 2에서 기본 실습 환경으로 소개된 **Google Colab (https://colab.research.google.com)**을 활용하면 복잡한 드라이버 설치 없이 클라우드 GPU(NVIDIA Tesla T4, 16GB VRAM)를 무료로 사용할 수 있습니다.
1. Colab 노트북 상단 메뉴에서 **`[런타임(Runtime)]` $\rightarrow$ `[런타임 유형 변경(Change runtime type)]`** 클릭.
2. 하드웨어 가속기(Hardware accelerator)를 **`T4 GPU`**로 선택하고 저장.
3. Colab에는 이미 CUDA가 연결된 TensorFlow, Keras, PyTorch, XGBoost가 모두 사전 설치되어 있으므로 위에서 설명한 `XGBClassifier(tree_method='hist', device='cuda')`와 `tf.keras` GPU 가속이 즉시 작동합니다.

