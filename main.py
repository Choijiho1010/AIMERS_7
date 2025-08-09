# main.py

import pandas as pd
import numpy as np
import argparse, itertools
import config
import os
import glob
import joblib
from tqdm import tqdm
# [수정] read_data와 _max_zero_run 함수를 preprocessing에서 임포트
from preprocessing import Preprocessor, read_data
from modeling import ModelTrainer
from utils import save_submission
from sklearn.metrics import mean_absolute_error

# ✅ validation 추가
from validation import (
    WeekAlignedValidator,
    build_week_aligned_blocks,
    load_calendar_blocks,
    last_5_weeks_block,
)
import utils  # smape

def main(args):
    
    if args.mode == 'train':
        print("--- 학습 모드 시작 ---")
        
        # 1. 데이터 불러오기 및 1차 전처리
        raw_df = read_data(config.TRAIN_CSV_PATH)
        
        preprocessor = Preprocessor(config)
        X_train, y_train = preprocessor.fit_transform(raw_df, is_train=True)

        # 2. Preprocessor 객체 저장
        os.makedirs(config.MODEL_DIR, exist_ok=True)
        joblib.dump(preprocessor, os.path.join(config.MODEL_DIR, 'preprocessor.pkl'))
        print("Preprocessor 객체가 저장되었습니다.")
        

        print(f"필터링 후 데이터 수: {len(X_train):,}")
        
        # 3. 모델 검증, 학습 및 저장
        print("\n--- 모델 학습 및 검증 시작 ---")
        trainer = ModelTrainer(config.MODELS, config.MODEL_DIR)

        # 3-1) 기본 holdout 검증
        trainer.validate(X_train, y_train)

        # 3-2) Week-aligned 검증 (공모전 동주차 + 마지막 5주)
        blocks = []
        blocks += load_calendar_blocks(config)      # 공모전 동주차 (config.VAL.calendar_blocks)
        blocks += [last_5_weeks_block(config)]      # 마지막 5주(4+1)
        if not blocks:                              # 혹시 비어 있으면 자동 생성
            blocks = build_week_aligned_blocks(config)

        #    - 각 블록마다 새 모델을 만들기 위한 팩토리 (trainer 내부 메서드 활용)
        factory = (lambda: trainer._get_model_instance(
            trainer.best_model_name,
            trainer.models[trainer.best_model_name]['model_params']
        ))

        validator = WeekAlignedValidator(
            model_factory=factory,
            gap_days=config.VAL.get("gap_days", 35),
            metric=utils.smape_leaderboard,
            progress=True
        )
        validator.run(X_train, y_train, blocks)

        # 3-3) 최종 전체 데이터로 재학습 후 저장
        trainer.fit_and_save(X_train, y_train)
        # 3-3) 최종 모델 저장
        trainer.fit_and_save(X_train, y_train)
        
        print("--- 학습 모드 완료 ---")

    elif args.mode == 'predict':
        print("--- 예측 모드 시작 ---")
        
        preprocessor = joblib.load(os.path.join(config.MODEL_DIR, 'preprocessor.pkl'))
        trainer = ModelTrainer(config.MODELS, config.MODEL_DIR)
        test_files = sorted(glob.glob(os.path.join(config.TEST_DIR_PATH, '*.csv')))
        
        all_predictions_list = []
        for test_file in tqdm(test_files, desc="Test 파일별 예측 진행"):
            test_df = read_data(test_file)
            
            X_test, _ = preprocessor.transform(test_df, is_train=False)
            
            # [수정] 필터링 로직을 제거하고 모든 test 데이터에 대해 예측 수행
            model_predictions = trainer.predict(X_test)
            
            # 예측값을 담을 배열 생성
            full_predictions = np.zeros((len(X_test), config.HORIZON))
            full_predictions[:] = model_predictions
            full_predictions = np.maximum(full_predictions, 1.0)

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