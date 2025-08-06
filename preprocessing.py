# preprocessing.py
import numpy as np
import pandas as pd
from tqdm import tqdm
import holidays
import itertools


class Preprocessor:
    def __init__(self, feature_settings):
        self.feature_settings = feature_settings
        self.all_items_ = None
    
    def fit(self, df):
        print("--- Preprocessor fit 시작 ---")
        self.all_items_ = df['영업장명_메뉴명'].unique()
        print(f"총 {len(self.all_items_)}개의 고유 품목을 학습했습니다.")
        print("--- Preprocessor fit 완료 ---")
        return self

    def transform(self, df):
        print("--- 1차 전처리(피처 생성) 시작 ---")
        processed_df = self._create_complete_dataframe(df)
        processed_df = self._create_calendar_features(processed_df)
        
        print("Lag/Rolling 피처 생성 중...")
        processed_df = self._create_lag_features(processed_df)
        processed_df = self._create_rolling_window_features(processed_df)

        print("--- 1차 전처리(피처 생성) 완료 ---")
        return processed_df

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

    def _create_complete_dataframe(self, df):
        """
        각 품목별로 데이터의 시작일부터 종료일까지 모든 날짜 행을 생성하고,
        판매 기록이 없는 날은 0으로 채워넣어 데이터의 연속성을 보장합니다.
        """
        # (품목, 날짜) 조합의 중복이 있을 경우를 대비한 안전장치
        df.drop_duplicates(subset=['영업장명_메뉴명', '영업일자'], keep='last', inplace=True)
        
        # '영업일자' 컬럼을 날짜 타입으로 변환
        df['영업일자'] = pd.to_datetime(df['영업일자'])
        
        # 데이터의 전체 시작일과 종료일 확인
        start_date, end_date = df['영업일자'].min(), df['영업일자'].max()
        all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # fit() 단계에서 학습한 전체 품목 리스트 사용
        if self.all_items_ is None:
            raise RuntimeError("Preprocessor가 아직 fit되지 않았습니다. fit을 먼저 실행하세요.")
        
        # (모든 품목 x 모든 날짜) 조합의 데이터 '틀' 생성
        multi_index = pd.MultiIndex.from_product([self.all_items_, all_dates], names=['영업장명_메뉴명', '영업일자'])
        
        # 원본 데이터를 '틀'에 맞게 재정렬
        df_reindexed = df.set_index(['영업장명_메뉴명', '영업일자']).reindex(multi_index).reset_index()
        
        # 빠진 날짜의 매출수량은 0으로 채움
        df_reindexed['매출수량'] = df_reindexed['매출수량'].fillna(0).astype(int)

        # 빠진 날짜의 영업장명, 메뉴명 정보 채우기
        split_data = df_reindexed['영업장명_메뉴명'].str.split('_', n=1, expand=True)
        df_reindexed['영업장명'], df_reindexed['메뉴명'] = split_data[0], split_data[1]
        
        # ffill()을 사용하여 빈 영업장명, 메뉴명을 바로 윗 행의 값으로 채움
        df_reindexed['영업장명'] = df_reindexed['영업장명'].ffill()
        df_reindexed['메뉴명'] = df_reindexed['메뉴명'].ffill()
        
        return df_reindexed

    def _create_calendar_features(self, df):
        # [수정] 공휴일 피처 생성 로직 추가
        kr_holidays = holidays.KR()
        df['is_holiday'] = df['영업일자'].apply(lambda x: 1 if x in kr_holidays or x.weekday() >= 5 else 0)
        df['holiday_block'] = (df['is_holiday'] != df['is_holiday'].shift()).cumsum()
        df['block_size'] = df.groupby('holiday_block')['is_holiday'].transform('size')
        df['consecutive_holidays'] = df['block_size'] * df['is_holiday']
        df.drop(columns=['holiday_block', 'block_size'], inplace=True)
        
        # 기존 날짜 피처 생성
        date_features_config = self.feature_settings.get('date_features', [])
        for feature in date_features_config:
            if feature == 'dayofweek': df['dayofweek'] = df['영업일자'].dt.dayofweek
            elif feature == 'month': df['month'] = df['영업일자'].dt.month
            elif feature == 'weekofyear': df['weekofyear'] = df['영업일자'].dt.isocalendar().week.astype(int)
        return df

    def _create_lag_features(self, df):
        lags = self.feature_settings.get('lags', [])
        for lag in lags:
            df[f'sales_lag_{lag}'] = df.groupby('영업장명_메뉴명')['매출수량'].shift(lag)
        return df

    def _create_rolling_window_features(self, df):
        windows = self.feature_settings.get('rolling_windows', [])
        for window in windows:
            rolling_series = df.groupby('영업장명_메뉴명')['매출수량'].shift(1).rolling(window)
            df[f'sales_rolling_mean_{window}'] = rolling_series.mean()
            df[f'sales_rolling_std_{window}'] = rolling_series.std()
        return df



def create_sliding_window_samples(processed_df, config):
    print("--- 2차 전처리(Tabular 변환) 시작 ---")
    
    horizon = config.OUTPUT_DAYS
    lag_cols = [f'sales_lag_{l}' for l in config.FEATURE_ENGINEERING['lags']]

    X_rows, y_rows = [], []

    for item_name, grp in tqdm(processed_df.groupby('영업장명_메뉴명'), desc="Sliding Window Generation"):
        threshold = config.ITEM_SPECIFIC_ZERO_RUN_THRESHOLDS.get(item_name, config.DEFAULT_ZERO_RUN_THRESHOLD)
        
        # [수정] for 루프 범위 +1
        for i in range(len(grp) - horizon + 1):
            features = grp.iloc[[i]]
            target = grp['매출수량'].iloc[i+1 : i+1+horizon]
            
            lag_values = features[lag_cols].values.flatten()
            if np.all(lag_values == 0): continue
            
            zero_runs = [len(list(g)) for v, g in itertools.groupby(lag_values) if v == 0]
            max_zero_run = max(zero_runs, default=0)
            
            if max_zero_run >= threshold: continue

            X_rows.append(features)
            y_rows.append(target.values)

    X = pd.concat(X_rows, ignore_index=True)
    y = pd.DataFrame(y_rows, columns=[f't+{h+1}' for h in range(horizon)])
    
    print(f"--- 2차 전처리(Tabular 변환) 완료 ---")
    print(f"생성된 X shape: {X.shape}, y shape: {y.shape}")
    
    return X, y