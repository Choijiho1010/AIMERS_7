# modeling.py

import os
import joblib # 모델 저장을 위한 라이브러리
import lightgbm as lgb
from tqdm import tqdm

class ModelTrainer:
    def __init__(self, model_settings, model_dir):
        self.settings = model_settings
        self.model_dir = model_dir
        self.models = {} # 학습된 모델 객체를 저장할 딕셔너리
        
        # 모델 저장 디렉토리 생성
        os.makedirs(self.model_dir, exist_ok=True)
        print(f"모델이 저장될 디렉토리: {self.model_dir}")

    def _get_model_instance(self, model_name):
        """설정에 따라 모델 인스턴스를 생성하는 내부 함수"""
        model_config = self.settings.get(model_name)
        if not model_config:
            raise ValueError(f"{model_name}에 대한 설정이 config.py에 없습니다.")

        if model_name == 'LightGBM':
            return lgb.LGBMRegressor(**model_config['model_params'])
        # elif model_name == 'XGBoost':
        #     return xgb.XGBRegressor(...) # 추후 확장 가능
        else:
            raise ValueError(f"지원하지 않는 모델입니다: {model_name}")

    def fit(self, X, y):
        """config에 명시된 모든 모델을 학습하고 저장합니다."""
        print("--- 모델 학습 시작 ---")
        for model_name in tqdm(self.settings.keys()):
            print(f"\n모델 학습 중: {model_name}")
            
            # 1. 모델 인스턴스 생성
            model = self._get_model_instance(model_name)
            
            # 2. 모델 학습
            model.fit(X, y)
            
            # 3. 학습된 모델을 딕셔너리와 파일로 저장
            self.models[model_name] = model
            model_path = os.path.join(self.model_dir, f"{model_name}.pkl")
            joblib.dump(model, model_path)
            print(f"{model_name} 모델이 {model_path}에 저장되었습니다.")
            
        print("--- 모든 모델 학습 완료 ---")
        return self.models

    def predict(self, X):
        """저장된 모든 모델을 불러와 예측을 수행합니다."""
        print("--- 예측 시작 ---")
        predictions = {}
        for model_name in tqdm(self.settings.keys()):
            print(f"\n모델 예측 중: {model_name}")
            
            # 1. 저장된 모델 불러오기
            model_path = os.path.join(self.model_dir, f"{model_name}.pkl")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"{model_path}에서 모델을 찾을 수 없습니다. fit을 먼저 실행하세요.")
            
            loaded_model = joblib.load(model_path)
            
            # 2. 예측 수행
            preds = loaded_model.predict(X)
            predictions[model_name] = preds
            
        print("--- 모든 모델 예측 완료 ---")
        return predictions