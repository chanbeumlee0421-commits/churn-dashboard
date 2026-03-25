import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="이탈 예측 대시보드", layout="wide")
st.title("🐾 동물병원 이탈 예측 대시보드")
st.caption("경보제약 동물의약품 | 기준일: 2026-03-24")

uploaded = st.file_uploader("Raw 엑셀 파일 업로드", type=["xlsx"])

if uploaded:
    df = pd.read_excel(uploaded, sheet_name="Raw")

    df_d = df[
        (df['유통'] == '신규처') &
        (df['거래처명'].notna())
    ].copy()
    df_d['매출일(배송완료일)'] = pd.to_datetime(df_d['매출일(배송완료일)'])

    ref_date = pd.Timestamp('2026-03-24')
    g = df_d.groupby('거래처명')

    features = pd.DataFrame({
        '첫구매일'   : g['매출일(배송완료일)'].min(),
        '마지막구매일': g['매출일(배송완료일)'].max(),
        '총구매횟수'  : g['매출일(배송완료일)'].count(),
        '구매제품수'  : g['품명요약2'].nunique(),
        '누적매출액'  : g['매출액(vat 제외)'].sum(),
        '담당자'     : g['담당자'].last(),
        '지역'       : g['지역1'].last(),
    })

    features['활동기간_일']  = (features['마지막구매일'] - features['첫구매일']).dt.days.fillna(0)
    features['미구매일수']   = (ref_date - features['마지막구매일']).dt.days.fillna(999)
    features['평균구매주기'] = features['활동기간_일'] / features['총구매횟수'].replace(0,1)
    features['거래처당매출'] = features['누적매출액'] / features['총구매횟수'].replace(0,1)
    features['이탈']        = (features['미구매일수'] >= 180).astype(int)

    le_mgr = LabelEncoder()
    le_reg = LabelEncoder()
    features['담당자_enc'] = le_mgr.fit_transform(features['담당자'].fillna('없음'))
    features['지역_enc']   = le_reg.fit_transform(features['지역'].fillna('없음'))

    feature_cols = ['총구매횟수', '구매제품수', '누적매출액', '활동기간_일',
                    '평균구매주기', '거래처당매출', '담당자_enc', '지역_enc']
    X = features[feature_cols].fillna(0)
    y = features['이탈']

    model = RandomForestClassifier(n_estimators=200, max_depth=6,
                                    random_state=42, class_weight='balanced')
    model.fit(X, y)
    features['이탈확률'] = model.predict_proba(X)[:, 1]
    features['위험등급'] = pd.cut(features['이탈확률'],
        bins=[0, 0.3, 0.5, 0.7, 1.0],
        labels=['🟢 안전', '🟡 주의', '🟠 위험', '🔴 긴급'])

    # ── 요약 지표
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("전체 거래처", f"{len(features)}개")
    col2.metric("🔴 긴급", f"{(features['위험등급']=='🔴 긴급').sum()}개")
    col3.metric("🟠 위험", f"{(features['위험등급']=='🟠 위험').sum()}개")
    col4.metric("🟢 안전", f"{(features['위험등급']=='🟢 안전').sum()}개")

    st.divider()

    # ── 필터
    col_f1, col_f2 = st.columns(2)
    mgr_list   = ['전체'] + sorted(features['담당자'].dropna().unique().tolist())
    grade_list = ['전체', '🔴 긴급', '🟠 위험', '🟡 주의', '🟢 안전']

    selected_mgr   = col_f1.selectbox("담당자 선택", mgr_list)
    selected_grade = col_f2.selectbox("위험등급 선택", grade_list)

    result = features.reset_index()[['거래처명', '담당자', '지역',
                                      '미구매일수', '총구매횟수', '구매제품수',
                                      '누적매출액', '이탈확률', '위험등급']].copy()

    if selected_mgr != '전체':
        result = result[result['담당자'] == selected_mgr]
    if selected_grade != '전체':
        result = result[result['위험등급'] == selected_grade]

    result = result.sort_values('이탈확률', ascending=False)
    result['이탈확률']  = (result['이탈확률'] * 100).round(1).astype(str) + '%'
    result['누적매출액'] = result['누적매출액'].apply(lambda x: f"{x:,.0f}원")

    st.subheader(f"거래처 목록 ({len(result)}개)")
    st.dataframe(result, use_container_width=True, hide_index=True)
