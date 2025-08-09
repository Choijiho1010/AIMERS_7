# validation.py
import pandas as pd
import numpy as np
import datetime as dt
from typing import List, Tuple, Callable
from tqdm import tqdm
import os
import utils
import config
import inspect

Date = dt.date
Block = Tuple[str, Date, Date, Date, Date]  # (label, inp_start, inp_end, tgt_start, tgt_end)

def _to_date(x) -> Date:
    if isinstance(x, pd.Timestamp): return x.date()
    return pd.to_datetime(x).date()

def _week_sunday(d: Date) -> Date:
    # Monday=0..Sunday=6 -> move back to Sunday
    return d - dt.timedelta(days=(d.weekday() + 1) % 7)

def build_week_aligned_blocks(cfg=config) -> List[Block]:
    """TRAIN_PERIOD에 맞춰 '일요일 시작' 4주+1주 자동 블록 생성"""
    tp = cfg.TRAIN_PERIOD
    start, end = _to_date(tp['start']), _to_date(tp['end'])
    cur = _week_sunday(start)
    if cur < start:
        cur += dt.timedelta(days=7)
    sundays = []
    d = cur
    while d + dt.timedelta(days=6) <= end:
        sundays.append(d); d += dt.timedelta(days=7)

    blocks: List[Block] = []
    for i in range(max(0, len(sundays) - 5)):             # 안전 범위
        inp_s = sundays[i]
        inp_e = sundays[i+4] - dt.timedelta(days=1)       # 28일
        tgt_s = sundays[i+4]
        tgt_e = sundays[i+5] - dt.timedelta(days=1)       # 7일
        if tgt_e <= end:
            blocks.append((f"AUTO_{i:02d}", inp_s, inp_e, tgt_s, tgt_e))
    return blocks

def last_5_weeks_block(cfg=config) -> Block:
    """TRAIN_PERIOD 끝 기준 마지막 5주(4+1) 블록"""
    end = _to_date(cfg.TRAIN_PERIOD['end'])
    tgt_s = _week_sunday(end)
    tgt_e = tgt_s + dt.timedelta(days=6)
    inp_e = tgt_s - dt.timedelta(days=1)
    inp_s = inp_e - dt.timedelta(days=27)
    return ("LAST5W", inp_s, inp_e, tgt_s, tgt_e)

def load_calendar_blocks(cfg=config) -> List[Block]:
    """VAL.calendar_blocks 수동 정의를 읽어 Block 리스트로 변환"""
    res = []
    for j, (is_, ie_, ts_, te_) in enumerate(cfg.VAL.get("calendar_blocks", [])):
        is_, ie_, ts_, te_ = map(_to_date, [is_, ie_, ts_, te_])
        res.append((f"CAL_{j:02d}", is_, ie_, ts_, te_))
    return res

def select_val_indices(X: pd.DataFrame, target_start: Date):
    """
    슬라이딩 윈도 샘플에서 검증 인덱스 선택.
    preprocessing2에서는 ref_date가 '타깃 시작일'이므로,
    ref_date == target_start로 잡아야 한다.
    """
    return X.index[X['ref_date'] == pd.to_datetime(target_start)].tolist()

def select_train_indices_with_gap(X: pd.DataFrame, input_start: Date, gap_days: int):
    """embargo: train은 input_start - gap_days 이전 ref_date까지만"""
    cutoff = pd.to_datetime(input_start) - pd.Timedelta(days=gap_days)
    return X.index[X['ref_date'] <= cutoff].tolist()

class WeekAlignedValidator:
    def __init__(self, model_factory, gap_days: int, metric, progress: bool = True, metric_kwargs: dict | None = None):
        self.model_factory = model_factory
        self.gap_days = gap_days
        self.metric = metric
        self.progress = progress
        self.metric_kwargs = metric_kwargs or {}
        self._metric_arity = len(inspect.signature(metric).parameters)

    def run(self, X: pd.DataFrame, y: pd.DataFrame, blocks: List[Block]):
        # 방어: 블록 형식 확인 (함수 잘못 넣는 실수 방지)
        for idx, b in enumerate(blocks):
            if not (isinstance(b, (list, tuple)) and len(b) == 5):
                raise TypeError(f"[blocks[{idx}]] invalid item: type={type(b)} val={b}")

        print("\n================ Week-Aligned Validation (sMAPE) ================")
        print("Label      Input(28d)                 Target(7d)                sMAPE    Note")
        print("--------------------------------------------------------------------------")

        dump_frames = [] # 시각화에 사용

        for label, inp_s, inp_e, tgt_s, tgt_e in blocks:
            # ✅ preprocessing2: ref_date == target_start 이므로 tgt_s로 선택
            val_idx = select_val_indices(X, tgt_s)
            if not val_idx:
                self._print_row(label, inp_s, inp_e, tgt_s, tgt_e, None, "no_val_rows"); continue

            tr_idx = select_train_indices_with_gap(X, inp_s, self.gap_days)
            if not tr_idx:
                self._print_row(label, inp_s, inp_e, tgt_s, tgt_e, None, "no_train_rows"); continue

            model = self.model_factory()  # 블록마다 새 모델
            X_tr = X.loc[tr_idx].drop(columns=['ref_date','is_filtered'], errors='ignore')
            y_tr = y.loc[tr_idx]
            X_va = X.loc[val_idx].drop(columns=['ref_date','is_filtered'], errors='ignore')
            y_va = y.loc[val_idx]

            try:
                model.fit(X_tr, y_tr)
                y_hat = model.predict(X_va)

                try:
                    if self._metric_arity >= 3:
                        # smape_leaderboard(X_meta, y_true, y_pred, **kwargs)
                        score = self.metric(X_va, y_va.values, y_hat, **self.metric_kwargs)
                    else:
                        # smape(y_true, y_pred, **kwargs)
                        score = self.metric(y_va.values, y_hat, **self.metric_kwargs)
                    self._print_row(label, inp_s, inp_e, tgt_s, tgt_e, score, "ok")
                except Exception as e:
                    self._print_row(label, inp_s, inp_e, tgt_s, tgt_e, None, f"ERR:{e}")
                
                # 시각화 용
                # ✅ 덤프 생성: meta는 'ref_date'가 필요하므로 X에서 직접 꺼냄
                X_meta = X.loc[val_idx, ['영업장명_메뉴명', 'ref_date']].reset_index(drop=True).copy()
                X_meta['store'] = X_meta['영업장명_메뉴명'].astype(str).str.split('_', n=1).str[0]
                X_meta['menu']  = X_meta['영업장명_메뉴명'].astype(str).str.split('_', n=1).str[1]
                X_meta['row_id'] = np.arange(len(X_meta))

                # y_true / y_pred → wide
                yv = pd.DataFrame(y_va.values, columns=[f"t+{i}" for i in range(1, y_hat.shape[1] + 1)])
                pr = pd.DataFrame(y_hat,     columns=[f"t+{i}" for i in range(1, y_hat.shape[1] + 1)])

                # row_id 부여
                yv['row_id'] = X_meta['row_id']
                pr['row_id'] = X_meta['row_id']

                # long 변환
                yv_long = yv.melt(id_vars='row_id', var_name='h', value_name='actual')
                pr_long = pr.melt(id_vars='row_id', var_name='h', value_name='pred')

                # ✅ row_id와 h로 머지 → 행 손실 없음
                df_dump = yv_long.merge(pr_long, on=['row_id','h']).merge(X_meta, on='row_id')

                df_dump['h'] = df_dump['h'].str.replace('t+', '', regex=False).astype(int)
                df_dump['pred_date'] = pd.to_datetime(df_dump['ref_date']) + pd.to_timedelta(df_dump['h'] - 1, unit='D')
                df_dump['split'] = label  # e.g., CAL_00..09 / LAST5W

                dump_frames.append(df_dump)

                
            except Exception as e:
                self._print_row(label, inp_s, inp_e, tgt_s, tgt_e, None, f"ERR:{e}")

        print("--------------------------------------------------------------------------\n")

        # 시각화용 :  파일로 저장
        if dump_frames:
            out = pd.concat(dump_frames, ignore_index=True)
            dump_dir = os.path.join(config.MODEL_DIR, "outputs")
            os.makedirs(dump_dir, exist_ok=True)
            out_path = os.path.join(dump_dir, "validation_predictions.csv")
            out.to_csv(out_path, index=False, encoding='utf-8-sig')
            print(f"[Dump] Validation predictions saved → {out_path}")

    @staticmethod
    def _fmt(d: Date) -> str:
        return d.strftime("%Y-%m-%d")

    def _print_row(self, label, ins, ine, ts, te, sc, note):
        sc_str = f"{sc:6.3f}" if isinstance(sc, float) else "  N/A "
        print(f"{label:8s}  {self._fmt(ins)} ~ {self._fmt(ine)}   {self._fmt(ts)} ~ {self._fmt(te)}   {sc_str}   {note}")
