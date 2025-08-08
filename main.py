# main.py

import pandas as pd
import numpy as np
import argparse, itertools
import config
import os
import glob
import joblib
import utils
from tqdm import tqdm
# [수정] read_data와 _max_zero_run 함수를 preprocessing에서 임포트
from preprocessing import Preprocessor, create_sliding_window_samples, read_data
from modeling import ModelTrainer
from sklearn.metrics import mean_absolute_error



def main(args):

    # ++++++++++++++++++ DB 연결 필요시 +++++++++++++++
    # engine = utils.connect_db()

    # # train 불러오기
    # sql = 'SELECT * FROM train'
    # train = utils.db2df(engine, sql)

    # # test 불러오기
    # test_list = []
    # for i in range(10):
    #     sql = f'SELECT * FROM TEST_0{i}'
    #     test = utils.db2df(engine, sql)
    #     test_list.append(test)
    
    # +++++++++++++++++++++++++++++++++++++++++++++
    
    if args.mode == 'train':
        print("--- 학습 모드 시작 ---")
        
        # 1. 데이터 불러오기 및 1차 전처리
        raw_df = read_data(config.TRAIN_CSV_PATH)
        
        preprocessor = Preprocessor(store_xy_map=config.STORE_XY, thr=config.THR)
        processed_df = preprocessor.fit_transform(raw_df)

        # 2. Preprocessor 객체 저장
        os.makedirs(config.MODEL_DIR, exist_ok=True)
        joblib.dump(preprocessor, os.path.join(config.MODEL_DIR, 'preprocessor.pkl'))
        print("Preprocessor 객체가 저장되었습니다.")
        
        # 3. 2차 전처리: Tabular 데이터셋 생성 (필터링 로직 포함)
        X_train, y_train = create_sliding_window_samples(
            processed_df,
            lags=config.LAGS,
            horizon=config.HORIZON,
            train_mode=True,
            thr=config.THR
        )

        print(f"필터링 후 데이터 수: {len(X_train):,}")
        
        # 4. 모델 검증, 학습 및 저장
        print("\n--- 모델 학습 및 검증 시작 ---")
        trainer = ModelTrainer(config.MODELS, config.MODEL_DIR)

        trainer.validate(X_train, y_train)
        trainer.fit_and_save(X_train, y_train)
        
        print("--- 학습 모드 완료 ---")

    elif args.mode == 'predict':
        print("--- 예측 모드 시작 ---")
        
        # [수정] Preprocessor 객체 로드만 수행하도록 수정
        preprocessor = joblib.load(os.path.join(config.MODEL_DIR, 'preprocessor.pkl'))
        trainer = ModelTrainer(config.MODELS, config.MODEL_DIR)
        test_files = sorted(glob.glob(os.path.join(config.TEST_DIR_PATH, '*.csv')))
        
        all_predictions_list = []
        for test_file in tqdm(test_files, desc="Test 파일별 예측 진행"):
            # [수정] read_data 함수를 import해서 사용
            test_df = read_data(test_file)
        # for test_df in test_list:
            
            X_test, _ = create_sliding_window_samples(
                preprocessor.transform(test_df),
                lags=config.LAGS,
                horizon=config.HORIZON,
                train_mode=False,
                thr=config.THR
            )

            filtered_mask = X_test['is_filtered']
            X_predict = X_test.loc[~filtered_mask].drop(columns=['is_filtered'])
            
            # [수정] 모델 예측 결과가 2차원 배열이 되도록 predict 함수 수정
            model_predictions = trainer.predict(X_predict)
            
            full_predictions = np.zeros((len(X_test), config.HORIZON))
            full_predictions[~filtered_mask] = model_predictions
            
            rows = []
            keys_info = X_test.drop_duplicates(subset='영업장명_메뉴명').reset_index()
            
            for i, p in enumerate(full_predictions):
                key = keys_info.loc[i, '영업장명_메뉴명']
                start_date = keys_info.loc[i, 'ref_date']
                
                for h in range(config.HORIZON):
                    date_label = f"{os.path.basename(test_file).split('.')[0]}+{h+1}일"
                    rows.append({
                        '영업장명_메뉴명': key,
                        '영업일자': date_label,
                        '매출수량': max(0.0, float(p[h]))
                    })
            
            pred_df = pd.DataFrame(rows)
            all_predictions_list.append(pred_df)

        full_pred = pd.concat(all_predictions_list, ignore_index=True)
        save_submission(full_pred)
        print("--- 예측 모드 완료 ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', help='실행 모드를 선택하세요 (train, predict)')
    args = parser.parse_args()
    
    main(args)