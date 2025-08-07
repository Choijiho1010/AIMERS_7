# preprocessing.py

import pandas as pd
import numpy as np
import itertools
from tqdm import tqdm
from datetime import timedelta
import config
from typing import List, Dict, Any, Tuple

# [수정] read_data 함수를 그대로 옮겨옴
def read_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {'영업일자', '영업장명_메뉴명', '매출수량'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"입력 파일에 필요한 컬럼이 없습니다: {missing}")
    df['영업일자'] = pd.to_datetime(df['영업일자'])
    if df['매출수량'].isna().any():
        df['매출수량'] = df['매출수량'].fillna(0)
    return df

class Preprocessor:
    def __init__(self, store_xy_map: dict, thr: int):
        self.store_xy_map = store_xy_map
        self.thr = thr
        self.all_items_ = None
        self.lags_ = [f'lag_{l}' for l in range(1, config.LAGS + 1)]

    def fit(self, df: pd.DataFrame):
        self.all_items_ = df['영업장명_메뉴명'].unique()
        return self

    def transform(self, df: pd.DataFrame):
        df = df.copy()
        
        # 1. 데이터 클리닝 및 날짜 피처 생성
        df.drop_duplicates(subset=['영업장명_메뉴명', '영업일자'], keep='last', inplace=True)
        df['영업일자'] = pd.to_datetime(df['영업일자'])
        df['dow'] = df['영업일자'].dt.dayofweek
        df['month'] = df['영업일자'].dt.month
        
        # 2. 영업장명/메뉴명 분리 및 공간(좌표) 피처 추가
        df = self._create_spatial_features(df)
        
        # 3. Lag 피처 추가
        for lag in range(1, config.LAGS + 1):
            df[f'lag_{lag}'] = df.groupby('영업장명_메뉴명')['매출수량'].shift(lag)

        return df

    def fit_transform(self, df: pd.DataFrame):
        self.fit(df)
        return self.transform(df)
    
    def _create_spatial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df[['영업장명', '메뉴명']] = df['영업장명_메뉴명'].str.split('_', n=1, expand=True)
        df['x'] = df['영업장명'].map(lambda s: self.store_xy_map.get(s, (np.nan, np.nan))[0])
        df['y'] = df['영업장명'].map(lambda s: self.store_xy_map.get(s, (np.nan, np.nan))[1])
        df = df.drop(columns=['영업장명', '메뉴명'])
        return df

# =================================================================
# 2차 전처리: Tabular 변환 (노트북 로직과 동일)
# =================================================================
def create_sliding_window_samples(
    df: pd.DataFrame,
    lags: int,
    horizon: int,
    train_mode: bool,
    thr: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    X_rows, y_rows = [], []

    for key, grp in tqdm(df.groupby('영업장명_메뉴명'), desc='window'):
        # [수정] _ensure_daily_continuity 함수로 보내기 전에 필요한 컬럼들을 추가
        grp_for_continuity = grp.copy()
        grp_for_continuity[['영업장명', '메뉴명']] = grp_for_continuity['영업장명_메뉴명'].str.split('_', n=1, expand=True)
        
        g = _ensure_daily_continuity(grp_for_continuity)
        sales = g['매출수량'].values
        dates = g['영업일자']
        
        # [수정] _ensure_daily_continuity 내부에서 'x', 'y' 컬럼을 생성하도록 변경
        # 이 부분을 Preprocessor 클래스에서 이미 처리했으므로, 해당 로직을 제거
        
        if len(sales) < lags:
            continue
        
        # 위치 피처도 연속성을 위해 ffill/bfill로 채워야 함
        g['x'] = g['영업장명'].map(lambda s: config.STORE_XY.get(s, (np.nan, np.nan))[0]).ffill().bfill()
        g['y'] = g['영업장명'].map(lambda s: config.STORE_XY.get(s, (np.nan, np.nan))[1]).ffill().bfill()


        if train_mode:
            for i in range(lags, len(sales) - horizon + 1):
                lag_block = sales[i - lags:i][::-1]
                target = sales[i:i + horizon]
                
                max_zero_run_val = _max_zero_run(lag_block)
                if max_zero_run_val >= thr:
                    continue
                
                ref_date = dates.iloc[i]
                
                X_row = {
                    '영업장명_메뉴명': key,
                    'dow': int(ref_date.dayofweek),
                    'month': int(ref_date.month),
                    'ref_date': ref_date,
                    'x': g.iloc[i]['x'],
                    'y': g.iloc[i]['y'],
                }
                for l in range(1, lags + 1):
                    X_row[f'lag_{l}'] = float(lag_block[l - 1])

                X_rows.append(X_row)
                y_rows.append(target)
        else:
            lag_block = sales[-lags:][::-1]
            last_date = dates.iloc[-1]
            ref_date = last_date + timedelta(days=1)
            
            max_zero_run_val = _max_zero_run(lag_block)
            is_filtered = (max_zero_run_val >= thr)

            X_row = {
                '영업장명_메뉴명': key,
                'dow': int(ref_date.dayofweek),
                'month': int(ref_date.month),
                'ref_date': ref_date,
                'x': g.iloc[-1]['x'],
                'y': g.iloc[-1]['y'],
                'is_filtered': is_filtered,
            }
            for l in range(1, lags + 1):
                X_row[f'lag_{l}'] = float(lag_block[l - 1])

            X_rows.append(X_row)
    
    X = pd.DataFrame(X_rows)

    if train_mode:
        y_cols = [f't+{h}' for h in range(1, horizon + 1)]
        y = pd.DataFrame(y_rows, columns=y_cols)
        assert 'ref_date' in X.columns
        return X, y
    else:
        return X, pd.DataFrame()

# 노트북에 있던 보조 함수들
def _ensure_daily_continuity(grp: pd.DataFrame) -> pd.DataFrame:
    g = grp.set_index('영업일자').sort_index()
    g = g.asfreq('D')
    for col in ['영업장명_메뉴명', '영업장명', '메뉴명']:
        g[col] = g[col].ffill().bfill()
    g['매출수량'] = g['매출수량'].fillna(0)
    g = g.reset_index().rename(columns={'index': '영업일자'})
    return g

def _max_zero_run(arr):
    runs = (len(list(g)) for v, g in itertools.groupby(arr) if v == 0.0)
    return max(runs, default=0)