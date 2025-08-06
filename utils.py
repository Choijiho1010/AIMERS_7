# utils.py

import os
import pandas as pd
import config

def save_submission(pred_df_wide):
    """
    예측 결과를 submission.csv 형식(wide format)에 맞게 저장하는 함수
    """
    # submission 디렉토리 생성
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)
    
    # sample submission 파일 로드해서 '영업일자' 컬럼만 가져오기
    sample_df = pd.read_csv(config.SUBMISSION_CSV_PATH)
    
    # 최종 제출 데이터프레임 생성
    # 예측값 앞에 '영업일자' 컬럼을 추가합니다.
    submission_df = pred_df_wide.copy()
    submission_df.insert(0, '영업일자', sample_df['영업일자'])
    
    # 파일로 저장
    submission_path = os.path.join(config.SUBMISSION_DIR, 'submission.csv')
    submission_df.to_csv(submission_path, index=False)
    
    print(f"제출 파일이 성공적으로 저장되었습니다: {submission_path}")
    
def get_feature_columns(df):
    """데이터프레임에서 모델 학습에 사용할 피처 컬럼 목록을 반환합니다."""
    return [col for col in df.columns if col not in ['영업일자', '영업장명_메뉴명', '영업장명', '메뉴명', '매출수량']]