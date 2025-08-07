# modeling.py

import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
import config

class ModelTrainer:
    def __init__(self, models: Dict, model_dir: str):
        self.models = models
        self.model_dir = model_dir
        self.best_model_name = 'XGBoost_MultiOutput'
        self.X_train = None
        self.y_train = None
        self.model = None

    def _get_model_instance(self, model_name: str, model_params: Dict) -> Any:
        if model_name == 'XGBoost_MultiOutput':
            categorical_features = config.CATEGORICAL_FEATURES
            numerical_features = [
                'x', 'y'
            ] + [f'lag_{l}' for l in range(1, config.LAGS + 1)]
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_features),
                    ('num', 'passthrough', numerical_features)
                ],
                remainder='drop'
            )
            base_model = XGBRegressor(**model_params)
            
            return Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('multi_output_regressor', MultiOutputRegressor(base_model, n_jobs=config.N_JOBS))
            ])
        else:
            raise ValueError(f"지원하지 않는 모델입니다: {model_name}")
    
    def _time_based_split(self, X: pd.DataFrame, y: pd.DataFrame, valid_days: int) -> Tuple[List[int], List[int]]:
        cutoff = X['ref_date'].max() - pd.Timedelta(days=valid_days)
        train_idx = X[X['ref_date'] <= cutoff].index.tolist()
        val_idx = X[X['ref_date'] > cutoff].index.tolist()
        return train_idx, val_idx

    def validate(self, X_raw: pd.DataFrame, y: pd.DataFrame):
        print("\n--- 교차 검증 시작 ---")
        model_name = self.best_model_name
        model_info = self.models[model_name]
        
        X_data = X_raw.drop(columns=['ref_date'], errors='ignore')
        y_data = y
        
        train_idx, val_idx = self._time_based_split(X_raw, y, valid_days=config.VALIDATION_SETTINGS['valid_days'])
        
        X_train, X_val = X_data.loc[train_idx], X_data.loc[val_idx]
        y_train, y_val = y_data.loc[train_idx], y_data.loc[val_idx]
        
        self.model = self._get_model_instance(model_name, model_info['model_params'])
        self.model.fit(X_train, y_train)

        preds = self.model.predict(X_val)
        rmse = ((preds - y_val.values) ** 2).mean(axis=0) ** 0.5
        
        print("[Validation] RMSE per horizon:", rmse.round(2))
        
        self.X_train = X_data.loc[train_idx]
        self.y_train = y_data.loc[train_idx]
        
        print("\n--- 교차 검증 완료 ---")

    def fit_and_save(self, X: pd.DataFrame, y: pd.DataFrame):
        print("\n--- 전체 데이터 학습 시작 ---")
        
        model_name = self.best_model_name
        model_info = self.models[model_name]
        
        if self.model is None:
            self.model = self._get_model_instance(model_name, model_info['model_params'])
            
        X_fit = X.drop(columns=['ref_date', 'is_filtered'], errors='ignore')
        
        self.model.fit(X_fit, y)
        
        os.makedirs(self.model_dir, exist_ok=True)
        model_path = os.path.join(self.model_dir, f'{model_name}.pkl')
        joblib.dump(self.model, model_path)
        print(f"모델 '{model_name}'이 '{model_path}'에 저장되었습니다.")
        print("--- 전체 데이터 학습 완료 ---")

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        model_name = self.best_model_name
        model_path = os.path.join(self.model_dir, f'{model_name}.pkl')
        model = joblib.load(model_path)
        
        X_predict = X_test.drop(columns=['ref_date', 'is_filtered'], errors='ignore')
        
        preds = model.predict(X_predict)
        
        if preds.ndim == 1:
            preds = preds.reshape(1, -1)
        
        return preds