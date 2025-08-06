# main.py
import pandas as pd
import numpy as np
import argparse
import config
import os
import glob
import joblib
from tqdm import tqdm
from preprocessing import Preprocessor, create_sliding_window_samples
from modeling import ModelTrainer
from utils import save_submission
from sklearn.metrics import mean_absolute_error

def main(args):
    
    if args.mode == 'train':
        print("--- 학습 모드 시작 ---")
        
        # 데이터 불러오기
        raw_df = pd.read_csv(config.TRAIN_CSV_PATH)
        
        # 전처리 preprocessing.py
        preprocessor = Preprocessor(config.FEATURE_ENGINEERING)
        processed_df = preprocessor.fit_transform(raw_df)

        # Preprocessor class 저장
        os.makedirs(config.MODEL_DIR, exist_ok=True)
        joblib.dump(preprocessor, os.path.join(config.MODEL_DIR, 'preprocessor.pkl'))
        print("Preprocessor 객체가 저장되었습니다.")
        
        # Tabular화
        X, y = create_sliding_window_samples(processed_df, config)
        
        # Modeling
        trainer = ModelTrainer(X, y, config.MODELS, config.MODEL_DIR)
        # Validation
        trainer.validate(config.CV_SETTINGS)      # CV_SETTINGS를 이용한 cross validation
        # Finalize
        trainer.fit()       # train 전부를 fitting  후 pkl 저장
        
        print("--- 학습 모드 완료 ---")

    elif args.mode == 'predict':
        print("--- 예측 모드 시작 ---")
        
        preprocessor_path = os.path.join(config.MODEL_DIR, 'preprocessor.pkl')
        preprocessor = joblib.load(preprocessor_path)

        # ==================== DEBUG 코드 2: 로드 후 확인 ====================
        if hasattr(preprocessor, 'all_items_') and preprocessor.all_items_ is not None:
            print(f"DEBUG (로드 후): Preprocessor 객체에 'all_items_' 속성이 존재합니다. 개수: {len(preprocessor.all_items_)}")
        else:
            print("DEBUG (로드 후): Preprocessor 객체에 'all_items_' 속성이 존재하지 않거나 비어있습니다!")
        # =================================================================

        trainer = ModelTrainer(config.MODELS, config.MODEL_DIR)
        test_files = sorted(glob.glob(os.path.join(config.TEST_DIR_PATH, '*.csv')))
        
        # ... (이하 예측 로직은 동일) ...
        daily_predictions = []
        for test_file in tqdm(test_files, desc="Test 파일별 예측 진행"):
            test_df = pd.read_csv(test_file)
            history_df = test_df.copy()

            for _ in range(config.OUTPUT_DAYS):
                processed_history = preprocessor.transform(history_df)
                latest_features_df = processed_history.tail(len(preprocessor.all_items_)) # 에러 발생 지점
                features = [col for col in latest_features_df.columns if col not in ['영업일자', '영업장명_메뉴명', '영업장명', '메뉴명', '매출수량']]
                
                prediction = trainer.predict(latest_features_df[features])['LightGBM']
                prediction[prediction < 0] = 0

                daily_predictions.append(prediction)

                next_date = latest_features_df['영업일자'].iloc[0] + pd.Timedelta(days=1)
                new_row = pd.DataFrame({
                    '영업일자': [next_date] * len(preprocessor.all_items_),
                    '영업장명_메뉴명': latest_features_df['영업장명_메뉴명'].values,
                    '매출수량': prediction
                })
                history_df = pd.concat([history_df, new_row]).iloc[len(preprocessor.all_items_):]

        print("--- 예측 결과 포맷 변환 중 ---")
        sample_df = pd.read_csv(config.SUBMISSION_CSV_PATH)
        item_columns = sample_df.columns[1:]
        final_array = np.vstack(daily_predictions)
        pred_df_wide = pd.DataFrame(final_array, columns=preprocessor.all_items_)
        pred_df_wide = pred_df_wide[item_columns]

        save_submission(pred_df_wide)
        print("--- 예측 모드 완료 ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', help='실행 모드를 선택하세요 (train, predict, ensemble)')
    args = parser.parse_args()
    
    main(args)
    
    # 디지털전략팀 강덕일 