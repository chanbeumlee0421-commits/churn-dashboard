import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="거래처 분석 대시보드", layout="wide")
st.title("🐾 거래처 성장 분석 대시보드")
st.caption("경보제약 동물의약품 | 기준일: 2026-03-24")

st.sidebar.markdown("""
### 그룹 분류 기준

| 그룹 | 기준 |
|---|---|
| 🚀 성장형 | 최근 90일 내 구매 + 최근 6개월 매출이 이전 6개월 대비 증가 |
| ✅ 안정형 | 최근 90일 내 구매 + 매출 변화 -10%~+10% + 구매 3회↑ |
| ⚠️ 위험형 | 90~180일 미구매 OR 최근 6개월 매출 10%↑ 감소 |
| 📉 저효율형 | 180일↑ 미구매 OR 전체 구매 2회 이하 |
| 🆕 신규 | 거래 시작 6개월 미만 (성장률 판단 불가) |

---
### 매출 트렌드 계산 방식
- 기준일(2026-03-24) 기준
- 최근 6개월 vs 이전 6개월 매출 비교
- 이전 6개월 매출이 0인 경우 → 신규로 분류
- +30% = 최근 6개월이 이전보다 30% 많음
- -20% = 최근 6개월이 이전보다 20% 적음

---
### 종합점수 (100점)
| 항목 | 가중치 |
|---|---|
| 매출 트렌드 | 35% |
| 누적매출액 | 25% |
| 구매횟수 | 20% |
| 제품다양성 | 10% |
| 최근성 | 10% |
""")

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
    cut6     = ref_date - pd.DateOffset(months=6)
    cut12    = ref_date - pd.DateOffset(months=12)

    # ── 주요제품 상위 3개 + 수량 ───────────────────────
    def top3_products(hosp_name):
        d   = df_d[df_d['거래처명'] == hosp_name]
        top = d.groupby('품명요약2')['매출수량'].sum().sort_values(ascending=False).head(3)
        return ' / '.join([f"{prod} {int(qty)}개" for prod, qty in top.items()])

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
    }).reset_index()

    features['활동기간_일']  = (features['마지막구매일'] - features['첫구매일']).dt.days.fillna(0)
    features['미구매일수']   = (ref_date - features['마지막구매일']).dt.days.fillna(999)
    features['평균구매주기'] = features['활동기간_일'] / features['총구매횟수'].replace(0,1)
    features['신규여부']     = (features['첫구매일'] >= cut6)

    # ── 최근 6개월 vs 이전 6개월 매출 비교 ─────────────
    def get_trend(hosp_name, is_new):
        if is_new:
            return None  # 신규는 계산 안 함
        d      = df_d[df_d['거래처명'] == hosp_name]
        recent = d[d['매출일(배송완료일)'] >  cut6]['매출액(vat 제외)'].sum()
        prev   = d[(d['매출일(배송완료일)'] > cut12) &
                   (d['매출일(배송완료일)'] <= cut6)]['매출액(vat 제외)'].sum()
        if prev == 0:
            return None  # 이전 6개월 매출 없으면 계산 제외
        return (recent - prev) / prev

    with st.spinner("거래처 패턴 분석 중..."):
        features['매출트렌드'] = features.apply(
            lambda r: get_trend(r['거래처명'], r['신규여부']), axis=1
        )
        features['주요제품'] = features['거래처명'].apply(top3_products)

    # ── 종합점수 ───────────────────────────────────────
    def normalize(series):
        mn, mx = series.min(), series.max()
        if mx == mn:
            return pd.Series([50.0]*len(series), index=series.index)
        return (series - mn) / (mx - mn) * 100

    trend_filled = features['매출트렌드'].fillna(0)
    features['종합점수'] = (
        normalize(trend_filled)               * 0.35 +
        normalize(features['누적매출액'])      * 0.25 +
        normalize(features['총구매횟수'])      * 0.20 +
        normalize(features['구매제품수'])      * 0.10 +
        normalize(-features['미구매일수'])     * 0.10
    ).round(1)

    # ── 그룹 분류 ──────────────────────────────────────
    def assign_group(row):
        inactive = row['미구매일수']
        trend    = row['매출트렌드']
        cnt      = row['총구매횟수']
        is_new   = row['신규여부']

        # 신규 (6개월 미만)
        if is_new:
            return '🆕 신규'
        # 저효율 (180일↑ 미구매 or 2회 이하)
        if inactive > 180 or cnt <= 2:
            return '📉 저효율형'
        # 성장형 (90일 내 구매 + 트렌드 양수)
        if inactive <= 90 and trend is not None and trend > 0:
            return '🚀 성장형'
        # 안정형 (90일 내 구매 + 트렌드 -10%~10% + 3회↑)
        if inactive <= 90 and cnt >= 3 and trend is not None and -0.1 <= trend <= 0.1:
            return '✅ 안정형'
        # 위험형 (90~180일 or 트렌드 -10% 이하)
        if inactive > 90 or (trend is not None and trend < -0.1):
            return '⚠️ 위험형'
        return '✅ 안정형'

    features['그룹'] = features.apply(assign_group, axis=1)

    # ── 요약 지표 ──────────────────────────────────────
    cols = st.columns(5)
    for col, label in zip(cols, ['🚀 성장형','✅ 안정형','⚠️ 위험형','📉 저효율형','🆕 신규']):
        col.metric(label, f"{(features['그룹']==label).sum()}개")

    st.divider()

    # ── 탭 구성 ────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["📋 거래처 목록", "🔍 그룹별 패턴", "📊 담당자 현황"])

    with tab1:
        col_f1, col_f2 = st.columns(2)
        mgr_list   = ['전체'] + sorted(features['담당자'].dropna().unique().tolist())
        group_list = ['전체','🚀 성장형','✅ 안정형','⚠️ 위험형','📉 저효율형','🆕 신규']
        selected_mgr   = col_f1.selectbox("담당자", mgr_list)
        selected_group = col_f2.selectbox("그룹",   group_list)

        result = features[[
            '거래처명','담당자','지역','그룹','종합점수',
            '총구매횟수','구매제품수','누적매출액',
            '매출트렌드','미구매일수','주요제품'
        ]].copy()

        if selected_mgr   != '전체':
            result = result[result['담당자'] == selected_mgr]
        if selected_group != '전체':
            result = result[result['그룹']   == selected_group]

        result = result.sort_values('종합점수', ascending=False)
        result['매출트렌드'] = result['매출트렌드'].apply(
            lambda x: f"+{x:.0%}" if x and x > 0
            else (f"{x:.0%}" if x is not None else "N/A")
        )
        result['누적매출액'] = result['누적매출액'].apply(lambda x: f"{x:,.0f}원")

        st.dataframe(result, use_container_width=True, hide_index=True)

    with tab2:
        selected_g = st.selectbox(
            "분석할 그룹 선택",
            ['🚀 성장형','✅ 안정형','⚠️ 위험형','📉 저효율형','🆕 신규']
        )
        grp = features[features['그룹'] == selected_g]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("거래처 수",     f"{len(grp)}개")
        c2.metric("평균 종합점수", f"{grp['종합점수'].mean():.1f}점")
        c3.metric("평균 구매횟수", f"{grp['총구매횟수'].mean():.1f}회")
        c4.metric("평균 제품수",   f"{grp['구매제품수'].mean():.1f}개")

        st.markdown("**📦 주요 구매 제품 Top 5 (수량 기준)**")
        prod_qty = (
            df_d[df_d['거래처명'].isin(grp['거래처명'])]
            .groupby('품명요약2')['매출수량'].sum()
            .sort_values(ascending=False).head(5)
        )
        st.bar_chart(prod_qty)

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("**🗺️ 지역 분포**")
            st.bar_chart(grp['지역'].value_counts())
        with col_r:
            st.markdown("**👤 담당자 분포**")
            st.bar_chart(grp['담당자'].value_counts())

        st.markdown("**📈 그룹 특징 요약**")
        st.dataframe(
            grp[['종합점수','총구매횟수','구매제품수',
                 '누적매출액','미구매일수']]
            .describe().round(1),
            use_container_width=True
        )

    with tab3:
        mgr_summary = features.groupby('담당자').agg(
            담당거래처=('거래처명',   'count'),
            성장형=('그룹', lambda x: (x=='🚀 성장형').sum()),
            안정형=('그룹', lambda x: (x=='✅ 안정형').sum()),
            위험형=('그룹', lambda x: (x=='⚠️ 위험형').sum()),
            저효율형=('그룹', lambda x: (x=='📉 저효율형').sum()),
            신규=('그룹',   lambda x: (x=='🆕 신규').sum()),
            평균종합점수=('종합점수', 'mean'),
            평균매출=('누적매출액',  'mean'),
        ).reset_index()
        mgr_summary['평균종합점수'] = mgr_summary['평균종합점수'].round(1)
        mgr_summary['평균매출']    = mgr_summary['평균매출'].apply(
            lambda x: f"{x:,.0f}원"
        )
        st.dataframe(mgr_summary, use_container_width=True, hide_index=True)
