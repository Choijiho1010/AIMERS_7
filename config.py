# config.py

import os
from datetime import timedelta

# --- 기본 설정 ---
RANDOM_STATE = 42
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
N_JOBS = 5 # cpu 코어 개수
USE_GPU = False # gpu 사용 여부

# --- 경로 설정 ---
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
SUBMISSION_DIR = os.path.join(OUTPUT_DIR, 'submissions')
DB_INFO_DIR = os.path.join('../',BASE_DIR)

# --- 데이터 경로 ---
TRAIN_CSV_PATH = os.path.join(DATA_DIR, 'train', 'train.csv')
TEST_DIR_PATH = os.path.join(DATA_DIR, 'test')
SAMPLE_SUBMISSION_CSV_PATH = os.path.join(DATA_DIR, 'submission', 'sample_submission.csv')

# --- 시계열 설정 (노트북과 동일) ---
LAGS = 28          # 과거 28일
HORIZON = 7        # 예측 7일
THR = 14           # 연속 0 매출 데이터 제거 기준

# --- 피처 설정 (노트북과 동일) ---
# 범주형 컬럼 (OneHotEncoder 사용)
CATEGORICAL_FEATURES = ['영업장명_메뉴명', 'dow', 'month']

# 영업장 위치(좌표) 정보
"""
연회장의 정확한 위치 확인 필요
유진이 초기 코드:
store_xy = {
    "담하"           : (620, 350),   # 기준점
    "미라시아"       : (620, 350),   # 동일 지점
    "느티나무 셀프BBQ": (500, 235),   # 북서쪽 165 px
    "포레스트릿"      : (770, 260),   # 북동쪽 175 px
    "연회장"          : (840, 380),   # 동-남쪽 220 px
    "카페테리아"      : (345, 300),   # 서쪽 275 px
    "화담숲주막"      : (920, 150),   # 북동쪽 360 px
    "화담숲카페"      : (920, 150),   # 주막과 동일 지점
    "연회장"        : (900, 350),    # 동쪽 평지
    "라그로타": (300, 300)
}
"""
STORE_XY = {
    "담하": (620, 350),
    "미라시아": (620, 350),
    "느티나무 셀프BBQ": (500, 235),
    "포레스트릿": (770, 260),
    "연회장": (900, 350),
    "카페테리아": (345, 300),
    "화담숲주막": (920, 150),
    "화담숲카페": (920, 150),
    "라그로타": (300, 300),
}

# --- 모델링 설정 ---
MODELS = {
    'XGBoost_MultiOutput': {
        'model_params': {
            'n_estimators': 2000,
            'learning_rate': 0.05,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'reg:squarederror',
            'random_state': RANDOM_STATE,
            'tree_method': 'gpu_hist' if USE_GPU else 'hist',
            'predictor': 'gpu_predictor' if USE_GPU else None, # GPU 사용 시
            'n_jobs': N_JOBS,
        }
    },
}

# --- 검증 설정 (노트북의 시간 기반 분할과 동일) ---
VALIDATION_SETTINGS = {
    'valid_days': 14  # 최근 14일을 검증용으로 사용
}