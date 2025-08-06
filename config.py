# config.py

import os

# --- 기본 설정 ---
RANDOM_STATE = 42
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- 경로 설정 ---

DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
MODEL_DIR = os.path.join(BASE_DIR, 'models') # 학습된 모델 저장 경로
SUBMISSION_DIR = os.path.join(OUTPUT_DIR, 'submissions') # 최종 결과물 저장 경로
PREDICTION_DIR = os.path.join(OUTPUT_DIR, 'predictions') # 개별 모델 예측 결과 저장 경로

# --- 데이터 경로 ---
TRAIN_CSV_PATH = os.path.join(DATA_DIR, 'train', 'train.csv')
TEST_DIR_PATH = os.path.join(DATA_DIR, 'test')
SUBMISSION_CSV_PATH = os.path.join(DATA_DIR, 'submission', 'sample_submission.csv')    # 저장할 submission.csv 이름

# --- 피처 엔지니어링 설정 ---
INPUT_DAYS = 28
OUTPUT_DAYS = 7

# 파생 변수와 관련 설정을 추가
FEATURE_ENGINEERING = {
    'lags': [7, 14, 21, 28],
    'rolling_windows': [7, 14, 28],
    'date_features': ['dayofweek', 'month', 'weekofyear', 'year']
}

SAMPLING = {
    'default_zero_run_threshold': 28,
    'item_specific_zero_run_thresholds': {
        "담하_공깃밥": 10,
        # ...
    }
}

# --- 검증(Validation) 설정 ---
CV_SETTINGS = {
    'strategy': 'TimeSeriesSplit',  # 사용할 교차 검증 전략 (e.g., 'KFold', 'StratifiedKFold')
    'n_splits': 5                   # 교차 검증 폴드(Fold) 수
}


# --- 모델링 설정 ---
MODELS = {
        'XGBoost': {
        'model_params': {
            'n_estimators': 2000,
            'learning_rate': 0.05,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'reg:squarederror',
            'random_state': RANDOM_STATE,
            'tree_method': 'hist', # CPU 사용을 위한 설정
            'n_jobs': 2,           # 사용자님 요청 반영
        }
    },
}

# --- 앙상블 설정 ---
ENSEMBLE = {
    'method': 'average'  # 'weighted_average', 'stacking' 등
}