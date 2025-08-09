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
    # 컬럼 존재 확인
    required = {'영업일자', '영업장명_메뉴명', '매출수량'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"입력 파일에 필요한 컬럼이 없습니다: {missing}")
    # 타입 캐스팅
    df['영업일자'] = pd.to_datetime(df['영업일자'])
    # 매출수량 결측은 0으로 (필요 시 조정)
    if df['매출수량'].isna().any():
        df['매출수량'] = df['매출수량'].fillna(0)

    # 음수는 0으로 보정
    df.loc[df['매출수량'] < 0, '매출수량'] = 0

    return df


class Preprocessor:
    def __init__(self, config):
        self.store_xy = config.STORE_XY
        self.thr = config.THR
        self.all_items_ = None
        self.lags_ = [f'lag_{l}' for l in range(1, config.LAGS + 1)]

    def fit(self, df: pd.DataFrame):
        self.all_items_ = df['영업장명_메뉴명'].unique()
        return self

    def transform(self, df: pd.DataFrame, is_train: bool = True):
        
        df = self._add_calendar_feats(df)
        X, y = self.build_lag_samples(df, config.LAGS, config.HORIZON, is_train)
        if is_train:
            X, y = self._apply_zero_run_filter(X, y)
        # 2. 영업장명/메뉴명 분리 및 공간(좌표) 피처 추가
        X = self._create_spatial_features(X)

        return X, y

    def fit_transform(self, df: pd.DataFrame, is_train: bool = True):
        self.fit(df)
        return self.transform(df, is_train)
    
    def _add_calendar_feats(self, df: pd.DataFrame):
        """요일, 월 등 캘린더 변수 추가"""
        df = df.copy()
        df['dow'] = df['영업일자'].dt.dayofweek
        df['month'] = df['영업일자'].dt.month
        return df
    
    def _create_spatial_features(self, X: pd.DataFrame) -> pd.DataFrame:
        X[['영업장명', '메뉴명']] = X['영업장명_메뉴명'].str.split('_', n=1, expand=True)
        X['x'] = X['영업장명'].map(lambda s: self.store_xy[s][0])
        X['y'] = X['영업장명'].map(lambda s: self.store_xy[s][1])
        X = X.drop(columns=['영업장명', '메뉴명'])
        return X
    
    def _apply_zero_run_filter(self, X, y):
        lag_cols = [f'lag_{i}' for i in range(1, 29)]  # lag_1 … lag_28
        
        def _max_zero_run(arr):
            runs = (len(list(g)) for v, g in itertools.groupby(arr) if v == 0.0)
            return max(runs, default=0)
        
        # 연속 0 길이
        X['max_zero_run'] = X[lag_cols].apply(_max_zero_run, axis=1)

        # 1) 연속 0 길이 계산 (이미 max_zero_run 컬럼 존재한다고 가정)
        mask_keep = X['max_zero_run'] < self.thr

        print(f"드롭 비율: {(~mask_keep).mean():.1%}")  # ≈ 18 %

        # 2) 학습·튜닝용 데이터
        X_fit = X.loc[mask_keep].drop(['max_zero_run'], axis=1)
        y_fit = y.loc[mask_keep]
        
        return X_fit, y_fit

    def ensure_daily_continuity(self, grp: pd.DataFrame) -> pd.DataFrame:
        """
        그룹(영업장명_메뉴명) 단위로 일자 연속성 확보.
        빠진 날짜는 매출수량=0으로 채움. 필요시 다른 보간법으로 교체.
        """
        g = grp.set_index('영업일자').sort_index()
        # 전체 기간에 대해 일 단위 리샘플
        g = g.asfreq('D')
        # 문자/카테고리 컬럼 보전
        g['영업장명_메뉴명'] = g['영업장명_메뉴명'].ffill().bfill()
        # 매출수량 결측은 0
        g['매출수량'] = g['매출수량'].fillna(0)
        g = g.reset_index().rename(columns={'index': '영업일자'})
        return g

    def build_lag_samples(
        self, 
        df: pd.DataFrame,
        lags: int,
        horizon: int,
        is_train: bool = True
    ):
        """
        Train 모드:
            각 그룹별로 (lags -> horizon) 윈도우를 모든 가능한 위치에서 생성.
            반환: X(DataFrame), y(DataFrame), meta(ref_date 포함)

        Test 모드:
            각 그룹별로 마지막 28일만 사용하여 1행 생성.
            반환: X_test(DataFrame), keys(list[(key, 예측시작일)])
        """
        X_rows, y_rows, keys = [], [], []

        for key, grp in tqdm(df.groupby('영업장명_메뉴명'), desc='window'):
            g = self.ensure_daily_continuity(grp)
            sales = g['매출수량'].values
            dates = g['영업일자']

            if len(sales) < lags:
                # 라그를 만들 수 없으면 스킵 (혹은 패딩)
                continue

            if is_train:
                # 전체 기간에서 슬라이딩
                for i in range(lags, len(sales) - horizon + 1):  # 수정 : 2024-06-09까지 생성하기 위해서
                    lag_block = sales[i - lags:i][::-1]     # 최신값이 lag_1
                    target = sales[i:i + horizon]          # t+1 ... t+H
                    ref_date = dates.iloc[i]               # 기준 시점(라그의 바로 다음 날)

                    X_row = {
                        '영업장명_메뉴명': key,
                        'dow': int(ref_date.dayofweek),
                        'month': int(ref_date.month),
                        'ref_date': ref_date,              # 검증 split용 (학습시 drop)
                    }
                    for l in range(1, lags + 1):
                        X_row[f'lag_{l}'] = float(lag_block[l - 1])

                    X_rows.append(X_row)
                    y_rows.append(target)
                    keys.append((key, ref_date))
            else:
                # 마지막 28일로 1행 생성 → 예측 시작일은 마지막 관측일 + 1
                lag_block = sales[-lags:][::-1]
                last_date = dates.iloc[-1]
                ref_date = last_date + timedelta(days=1)
                
                # [수정] ref_date를 X_row에 추가
                X_row = {
                    '영업장명_메뉴명': key,
                    'dow': int(last_date.dayofweek),
                    'month': int(last_date.month),
                    'ref_date': ref_date,
                }
                for l in range(1, lags + 1):
                    X_row[f'lag_{l}'] = float(lag_block[l - 1])

                X_rows.append(X_row)
                keys.append((key, last_date + timedelta(days=1)))

        X = pd.DataFrame(X_rows)

        if is_train:
            y_cols = [f't+{h}' for h in range(1, horizon + 1)]
            y = pd.DataFrame(y_rows, columns=y_cols)
            # ref_date가 없는 행은 제거(안전장치)
            assert 'ref_date' in X.columns
            return X, y
        else:
            return X, keys




if __name__ == "__main__":
    import config
    raw_df = read_data(config.TRAIN_CSV_PATH)
    preprocessor = Preprocessor(config)
    df = preprocessor.fit_transform(raw_df)
