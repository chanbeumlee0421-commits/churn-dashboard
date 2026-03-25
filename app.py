import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from itertools import combinations
from collections import defaultdict

st.set_page_config(page_title="거래처 분석 대시보드", layout="wide")
st.title("🐾 거래처 성장 분석 대시보드")
st.caption("경보제약 동물의약품 | 기준일: 2026-03-24")

uploaded = st.file_uploader("Raw 엑셀 파일 업로드", type=["xlsx"])

if uploaded:
    df = pd.read_excel(uploaded, sheet_name="Raw")
    df_d = df[
        (df['거래구분'] == '신규처') &
        (df['거래처명'].notna())
    ].copy()
    df_d['매출일(배송완료일)'] = pd.to_datetime(df_d['매출일(배송완료일)'], errors='coerce')
    df_d = df_d[df_d['매출일(배송완료일)'].notna()]
    ref_date = pd.Timestamp('2026-03-24')

    # ── 피처 생성 ──────────────────────────────────────
    g = df_d.groupby('거래처명')

    features = pd.DataFrame({
        '첫구매일'   : g['매출일(배송완료일)'].min(),
        '마지막구매일': g['매출일(배송완료일)'].max(),
        '총구매횟수'  : g['매출일(배송완료일)'].count(),
        '구매제품수'  : g['품명요약2'].nunique(),
        '누적매출액'  : g['매출액(vat 제외)'].sum(),
        '담당자'     : g['담당자'].last(),
        '지역'       : g['지역1'].last(),
        '주요제품'   : g['품명요약2'].agg(lambda x: x.value_counts().index[0]),
        '수의사몰'   : g['수의사몰 가입'].last(),
    }).reset_index()

    features['활동기간_일']  = (features['마지막구매일'] - features['첫구매일']).dt.days.fillna(0)
    features['미구매일수']   = (ref_date - features['마지막구매일']).dt.days.fillna(999)
    features['평균구매주기'] = (features['활동기간_일'] / features['총구매횟수'].replace(0,1))
    features['거래처당매출'] = (features['누적매출액'] / features['총구매횟수'].replace(0,1))

    # 성장 트렌드: 전반기 vs 후반기 매출 비교
    mid_date = features['첫구매일'] + (features['마지막구매일'] - features['첫구매일']) / 2

    def get_trend(row):
        hosp = row['거래처명']
        mid  = row['첫구매일'] + (row['마지막구매일'] - row['첫구매일']) / 2
        d    = df_d[df_d['거래처명'] == hosp]
        early = d[d['매출일(배송완료일)'] <= mid]['매출액(vat 제외)'].sum()
        late  = d[d['매출일(배송완료일)'] >  mid]['매출액(vat 제외)'].sum()
        if early == 0: return 0
        return (late - early) / early

    with st.spinner("거래처 패턴 분석 중..."):
        features['매출트렌드'] = features.apply(get_trend, axis=1)

    # ── 클러스터링 ─────────────────────────────────────
    cluster_cols = ['총구매횟수', '구매제품수', '누적매출액',
                    '평균구매주기', '거래처당매출', '매출트렌드', '미구매일수']

    X = features[cluster_cols].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    features['클러스터'] = kmeans.fit_predict(X_scaled)

    # 클러스터 자동 라벨링
    summary = features.groupby('클러스터').agg(
        매출트렌드=('매출트렌드', 'mean'),
        누적매출액=('누적매출액', 'mean'),
        미구매일수=('미구매일수', 'mean'),
        구매제품수=('구매제품수', 'mean'),
    )

    def label_cluster(row):
        if row['매출트렌드'] > 0.2 and row['미구매일수'] < 180:
            return '🚀 성장형'
        elif row['미구매일수'] < 90 and row['누적매출액'] > summary['누적매출액'].mean():
            return '✅ 안정형'
        elif row['미구매일수'] > 270:
            return '📉 저효율형'
        else:
            return '⚠️ 위험형'

    cluster_labels = {i: label_cluster(summary.loc[i]) for i in summary.index}
    features['그룹'] = features['클러스터'].map(cluster_labels)

    # ── 요약 지표 ──────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    for col, label in zip([col1,col2,col3,col4],
                           ['🚀 성장형','✅ 안정형','⚠️ 위험형','📉 저효율형']):
        cnt = (features['그룹'] == label).sum()
        col.metric(label, f"{cnt}개")

    st.divider()

    # ── 탭 구성 ────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["📋 거래처 목록", "🔍 그룹별 패턴", "📊 담당자 현황"])

    with tab1:
        col_f1, col_f2 = st.columns(2)
        mgr_list   = ['전체'] + sorted(features['담당자'].dropna().unique().tolist())
        group_list = ['전체', '🚀 성장형', '✅ 안정형', '⚠️ 위험형', '📉 저효율형']
        selected_mgr   = col_f1.selectbox("담당자", mgr_list)
        selected_group = col_f2.selectbox("그룹", group_list)

        result = features[['거래처명','담당자','지역','그룹','총구매횟수',
                            '구매제품수','누적매출액','매출트렌드',
                            '미구매일수','주요제품','수의사몰']].copy()

        if selected_mgr   != '전체': result = result[result['담당자'] == selected_mgr]
        if selected_group != '전체': result = result[result['그룹']   == selected_group]

        result = result.sort_values('누적매출액', ascending=False)
        result['매출트렌드'] = result['매출트렌드'].apply(lambda x: f"+{x:.0%}" if x > 0 else f"{x:.0%}")
        result['누적매출액'] = result['누적매출액'].apply(lambda x: f"{x:,.0f}원")

        st.dataframe(result, use_container_width=True, hide_index=True)

    with tab2:
        selected_g = st.selectbox("분석할 그룹 선택",
                                   ['🚀 성장형','✅ 안정형','⚠️ 위험형','📉 저효율형'])
        grp = features[features['그룹'] == selected_g]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("거래처 수",    f"{len(grp)}개")
        c2.metric("평균 매출액",  f"{grp['누적매출액'].mean():,.0f}원")
        c3.metric("평균 구매횟수", f"{grp['총구매횟수'].mean():.1f}회")
        c4.metric("평균 제품수",  f"{grp['구매제품수'].mean():.1f}개")

        st.markdown("**주요 구매 제품**")
        prod_cnt = df_d[df_d['거래처명'].isin(grp['거래처명'])]['품명요약2'].value_counts().head(5)
        st.bar_chart(prod_cnt)

        st.markdown("**지역 분포**")
        region_cnt = grp['지역'].value_counts()
        st.bar_chart(region_cnt)

        st.markdown("**담당자 분포**")
        mgr_cnt = grp['담당자'].value_counts()
        st.bar_chart(mgr_cnt)

    with tab3:
        mgr_summary = features.groupby('담당자').agg(
            담당거래처=('거래처명',  'count'),
            성장형=('그룹', lambda x: (x=='🚀 성장형').sum()),
            안정형=('그룹', lambda x: (x=='✅ 안정형').sum()),
            위험형=('그룹', lambda x: (x=='⚠️ 위험형').sum()),
            저효율형=('그룹', lambda x: (x=='📉 저효율형').sum()),
            평균매출=('누적매출액', 'mean'),
        ).reset_index()
        mgr_summary['평균매출'] = mgr_summary['평균매출'].apply(lambda x: f"{x:,.0f}원")
        st.dataframe(mgr_summary, use_container_width=True, hide_index=True)
