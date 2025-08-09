# utils.py

import pandas as pd
import numpy as np
import os
import config
from sqlalchemy import create_engine, text
import json


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


def connect_db():

    with open("db_info.json") as f:
        db_info = json.load(f)

    user = db_info["user"]
    password = db_info["password"]
    host = db_info["host"]
    port = db_info["port"]
    schema = db_info["schema"]

    engine = None

    try:
        engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}:{port}/{schema}")
    except Exception as e:
        print(f'fail to connect db:{e}') 


    return engine
    

def df2db(engine, df, name, if_exist = 'replace', index = False):
    df.to_sql(
        name = name,             # RDS에 생성할 테이블 이름
        con = engine,
        if_exist= if_exist,     # 'replace': 기존 테이블 덮어쓰기, 'append': 이어쓰기
        index = index           # DataFrame의 인덱스를 테이블에 포함하지 않음
    )
    print("DataFrame successfully uploaded to RDS.")


def db2df(engine, sql):
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn)

    return df



def smape(y_true, y_pred, eps=1e-9):
    """
    Symmetric MAPE (sMAPE)
    - y_true, y_pred: numpy array 또는 pandas DataFrame (shape 동일)
    - 계산식: mean( 200 * |y - ŷ| / (|y| + |ŷ|) )
    - 분모가 0 근처일 때를 대비해 eps로 하한을 둠
    - 반환: float (평균 sMAPE)
    """

    # DataFrame이 들어오면 값만 추출
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.values
    if isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.values

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    num = np.abs(y_true - y_pred)
    den = np.clip(np.abs(y_true) + np.abs(y_pred), eps, None)

    return float(np.mean(200.0 * num / den))

def smape_point(a, p, eps=1e-9):
    a = np.asarray(a, dtype=float)
    p = np.asarray(p, dtype=float)
    den = np.clip(np.abs(a) + np.abs(p), eps, None)
    return 200.0 * np.abs(a - p) / den

def smape_leaderboard(
    X_meta: pd.DataFrame,
    y_true,
    y_pred,
    store_col: str = '영업장명_메뉴명',
    eps: float = 1e-9,
    store_weights: dict | None = None,   # ✅ 가게별 가중치 딕셔너리
) -> float:
    """
    리더보드 스타일 sMAPE (가게 가중 포함):
      - 실제 A=0인 날짜 제외
      - (store,item) 평균 → store 평균 → store 가중 평균
    기본 가중치: 균등. store_weights로 개별 가중치 오버라이드 가능.
    """
    # 1) 가게 추출 (영업장_메뉴명 --> 영업장)
    meta = X_meta[[store_col]].copy()
    meta['store'] = meta[store_col].astype(str).str.split('_', n=1).str[0].str.strip()
    meta['row_id'] = np.arange(len(meta))

    # 2) y를 long으로
    yt = pd.DataFrame(y_true).copy()
    yp = pd.DataFrame(y_pred).copy()
    yt['row_id'] = yp['row_id'] = np.arange(len(meta))
    yt = yt.melt(id_vars='row_id', var_name='h', value_name='A')
    yp = yp.melt(id_vars='row_id', var_name='h', value_name='P')
    df = yt.merge(yp, on=['row_id','h']).merge(meta[['row_id','store',store_col]], on='row_id')

    # 3) 실제=0 제외
    df = df[df['A'] != 0].copy()
    if df.empty:
        return np.nan

    # 4) 포인트 sMAPE
    df['smape'] = smape_point(df['A'], df['P'], eps=eps)

    # 5) 품목 → 가게 평균
    item_mean = df.groupby(['store', store_col])['smape'].mean()
    store_mean = item_mean.groupby('store').mean()                # index = store

    # 6) ✅ 가게 가중 평균 (미라시아/담하 2x)
    default_w = {'미라시아': 2.0, '담하': 2.0}                   # <- 원하는 가중치
    if store_weights:                                             # 호출 시 오버라이드 가능
        default_w.update(store_weights)
    w = store_mean.index.to_series().map(lambda s: default_w.get(s, 1.0)).astype(float)

    score = float((store_mean * w).sum() / w.sum())
    return score