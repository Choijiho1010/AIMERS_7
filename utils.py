# utils.py

import pandas as pd
import numpy as np
import os
import config

def save_submission(predictions_df: pd.DataFrame, file_name: str = "submission.csv"):
    """
    예측 결과를 submission.csv 형식에 맞게 저장하는 함수
    """
    sample_submission = pd.read_csv(config.SAMPLE_SUBMISSION_CSV_PATH)
    
    pred_dict = dict(zip(
        zip(predictions_df['영업일자'], predictions_df['영업장명_메뉴명']),
        predictions_df['매출수량']
    ))

    final_df = sample_submission.copy()

    for row_idx in final_df.index:
        date = final_df.loc[row_idx, '영업일자']
        for col in final_df.columns[1:]:
            final_df[col] = final_df[col].astype(float)
            final_df.loc[row_idx, col] = pred_dict.get((date, col), 0)

    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)
    out_path = os.path.join(config.SUBMISSION_DIR, file_name)
    final_df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"제출 파일이 '{out_path}'에 저장되었습니다.")