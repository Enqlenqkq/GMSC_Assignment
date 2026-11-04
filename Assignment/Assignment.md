Give Me Some Credit 데이터셋 (출처: Kaggle Competition, 2011)
https://www.kaggle.com/competitions/GiveMeSomeCredit/overview

• 목적: 개인 고객의 재무 데이터를 바탕으로 2년 이내에 심각한 재정적 곤경(financial distress, default)에 빠질 확률을 예측하는 이진 분류 문제
• 분야: 신용평가(Credit Scoring), 리스크 관리, 불균형 데이터(Imbalanced) 분류 기법 연구
• 데이터형태: (150000, 9) → target(Y): SeriousDlqin2yr, features(x): 그 외 8개 변수
기존 데이터가 결측치 등을 포함하여 데이터 전처리 완료된 데이터, NOTE 논문에 feature 설명있음
• 과제: Lab1 – Lab6 진행해온 학습을 바탕으로 새로운 데이터셋을 활용하여
AI Modeling 및 Predictive Analytics 를 진행 후, 7분 정도의 발표 분량으로 PPT 작성
• 평가: Week13 이전까지 제출(시간지연 시 50% 감점 페널티부과)
Week13 및 Week14 에 과제 발표
[참고문헌]
• Han, Seongil, Haemin Jung, Paul D. Yoo, Alessandro Provetti, and Andrea Cali 
NOTE: Non-parametric Oversampling Technique for Explainable credit scoring → AUROC: 98.37 
Nature Scientific Reports 14, no. 1 (2024): 26070. [link] (Random Forest) 
• Han, Seongil, and Haemin Jung
NATE: Non-pArameTric approach for Explainable credit scoring on imbalanced class → AUROC: 96.49 
PLOS One 19, no. 12 (2024): e0316454. [link] (Gradient Boosting)