# app.py
import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="예측 성능 대시보드", layout="wide")

# -----------------------
# 유틸≠
# -----------------------
@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # 타입 정리
    for c in ['ref_date', 'pred_date']:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors='coerce')
    return df

def smape_point(a, p, eps=1e-9, scale_200=True):
    a = np.asarray(a, dtype=float); p = np.asarray(p, dtype=float)
    den = np.clip(np.abs(a) + np.abs(p), eps, None)
    s = 2.0 * np.abs(a - p) / den  # 0~2
    return (s * 100) if scale_200 else s  # 0~200 or 0~2

# -----------------------
# 사이드바: 데이터 선택
# -----------------------
# ---- 데이터 불러오기 ----
base_out = os.path.join("models", "outputs")
path_hold = os.path.join(base_out, "holdout_predictions.csv")
path_val  = os.path.join(base_out, "validation_predictions.csv")

dfs = []
if os.path.exists(path_hold):
    dfs.append(load_csv(path_hold))
if os.path.exists(path_val):
    dfs.append(load_csv(path_val))

uploaded = st.sidebar.file_uploader("추가 CSV 업로드(+합치기)", type=["csv"], accept_multiple_files=True)
if uploaded:
    for up in uploaded:
        dfs.append(load_csv(up))

if not dfs:
    st.warning("CSV를 찾을 수 없습니다. (holdout/validation) 경로를 확인하거나 업로드하세요.")
    st.stop()

df = pd.concat(dfs, ignore_index=True)

# ---- 컬럼/타입 정리 ----
for c in ['ref_date','pred_date']:
    if c in df.columns: df[c] = pd.to_datetime(df[c], errors='coerce')

# 필수 컬럼 확인
required = {'영업장명_메뉴명','store','menu','ref_date','h','pred_date','actual','pred','split'}
missing = required - set(df.columns)
if missing:
    st.error(f"필수 컬럼 누락: {missing}")
    st.stop()

# ---- 사이드바 필터 ----
st.sidebar.header("필터")
scale_200 = st.sidebar.selectbox("sMAPE 스케일", ["0~200", "0~2"]) == "0~200"

splits = sorted(df['split'].dropna().unique().tolist())  # ['holdout','CAL_00',...]
split_sel = st.sidebar.multiselect("Split 선택", splits, default=splits)
df = df[df['split'].isin(split_sel)].copy()

min_d, max_d = df['pred_date'].min(), df['pred_date'].max()
date_range = st.sidebar.date_input("날짜 범위", (min_d, max_d))
if isinstance(date_range, tuple):
    start_d, end_d = date_range
else:
    start_d, end_d = min_d, max_d
mask_date = (df['pred_date'].dt.date >= start_d) & (df['pred_date'].dt.date <= end_d)
df = df.loc[mask_date].copy()

# ---- 지표 계산 & 시각화 (기존과 동일) ----
df['smape'] = smape_point(df['actual'], df['pred'], scale_200=scale_200)
df['ae'] = np.abs(df['actual'] - df['pred'])

st.title("매장·메뉴 예측 성능 대시보드 (Holdout + Validation)")

# -----------------------
# 1) 매장/메뉴별 손실 랭킹
# -----------------------
st.subheader("손실 랭킹 (매장별 · 메뉴별)")

col1, col2 = st.columns(2)

with col1:
    metric_for_store = st.selectbox("매장 랭킹 지표", ["sMAPE", "MAE"], key="m_store")
    agg_col = 'smape' if metric_for_store == "sMAPE" else 'ae'
    store_rank = df.groupby('store')[agg_col].mean().sort_values(ascending=False).reset_index()
    fig = px.bar(store_rank.head(30), x=agg_col, y='store', orientation='h',
                 title=f"매장별 평균 {metric_for_store} (Top 30)")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    metric_for_item = st.selectbox("메뉴(매장-메뉴) 랭킹 지표", ["sMAPE", "MAE"], key="m_item")
    agg_col2 = 'smape' if metric_for_item == "sMAPE" else 'ae'
    item_rank = df.groupby('영업장명_메뉴명')[agg_col2].mean().sort_values(ascending=False).reset_index()
    fig2 = px.bar(item_rank.head(30), x=agg_col2, y='영업장명_메뉴명', orientation='h',
                  title=f"매장-메뉴별 평균 {metric_for_item} (Top 30)")
    st.plotly_chart(fig2, use_container_width=True)

# -----------------------
# 2) 필터 & 개별 시계열 비교
# -----------------------
st.subheader("개별 매장·메뉴 시계열")

# 1) 매장 선택: '전체 매장' 추가
stores = ["전체 매장"] + sorted(df['store'].dropna().unique().tolist())
store_sel = st.selectbox("매장 선택", stores)

# 2) 매장 필터 적용
if store_sel == "전체 매장":
    df_store = df.copy()
else:
    df_store = df[df['store'] == store_sel].copy()

# 3) 메뉴 선택: '전체 메뉴' 추가 (현재 매장 필터 기준)
menus = ["전체 메뉴"] + sorted(df_store['menu'].dropna().unique().tolist())
menu_sel = st.selectbox("메뉴 선택", menus)

# 4) 메뉴 필터 적용
if menu_sel == "전체 메뉴":
    sel = df_store.copy()
else:
    sel = df_store[df_store['menu'] == menu_sel].copy()

if sel.empty:
    st.info("선택한 매장/메뉴에 해당하는 데이터가 없습니다.")
else:
    # 여러 항목이 섞일 수 있으므로 집계 방법 선택
    agg_method = st.radio("여러 항목 선택 시 집계 방식", ["합계(sum)", "평균(mean)"], index=0, horizontal=True)
    agg_func = "sum" if agg_method.startswith("합계") else "mean"

    # 일자별 집계(실제/예측)
    daily = sel.groupby('pred_date')[['actual', 'pred']].agg(agg_func).reset_index()

    # 손실 계산
    daily['ae'] = np.abs(daily['actual'] - daily['pred'])
    daily['smape'] = smape_point(daily['actual'], daily['pred'], scale_200=scale_200)

    # 제목 만들기
    title_store = store_sel
    title_menu = menu_sel
    if store_sel == "전체 매장":  title_store = "전체 매장"
    if menu_sel  == "전체 메뉴":  title_menu  = "전체 메뉴"

    # 라인 차트: 실제 vs 예측
    fig_ts = px.line(daily, x='pred_date', y=['actual','pred'],
                     title=f"[{title_store} - {title_menu}] 실제 vs 예측 (일자별, {agg_method})")
    st.plotly_chart(fig_ts, use_container_width=True)

    # 손실 추이 (선택된 스케일에 맞춰 sMAPE 또는 MAE)
    loss_metric_name = st.selectbox("손실 지표", ["sMAPE", "MAE"], index=0)
    ycol = 'smape' if loss_metric_name == "sMAPE" else 'ae'
    fig_loss = px.bar(daily, x='pred_date', y=ycol,
                      title=f"손실 추이 ({loss_metric_name}, {agg_method})")
    st.plotly_chart(fig_loss, use_container_width=True)

    with st.expander("원본 표 보기"):
        st.dataframe(sel.sort_values(['pred_date','h']))