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
| 🚀 성장형 | 분기 4개↑ + 최근 2분기 > 직전 2분기 10%↑ + 미구매일수 < 평균주기×1.5 |
| ✅ 안정형 | 거래 1년↑ + 구매 6회↑ + 누적매출 210만원↑ + 미구매일수 < 평균주기×1.5 |
| ⚠️ 위험형 | 구매 3회↑ + 미구매일수 > 평균주기×1.5 |
| 📉 저효율형 | 구매 2회 이하 OR 365일↑ 미구매 |
| 🆕 신규 | 첫거래 6개월 미만 + 구매 4회 이하 |

---
### 주요 지표 설명

**성장률**
직전 2분기 평균 대비 최근 2분기 평균 변화율
분기 4개 미만이면 N/A

**평균구매주기**
전체 거래 기간 ÷ 구매횟수

**종합점수 (100점)**
| 항목 | 가중치 |
|---|---|
| 성장률 | 35% |
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
    df_d = df_d.sort_values(['거래처명', '매출일(배송완료일)'])
    df_d['분기'] = df_d['매출일(배송완료일)'].dt.to_period('Q')
    ref_date = pd.Timestamp('2026-03-24')

    qtr_sales = df_d.groupby(['거래처명','분기'])['매출액(vat 제외)'].sum().reset_index()

    def get_growth(hosp_name):
        q = qtr_sales[qtr_sales['거래처명']==hosp_name].sort_values('분기')
        if len(q) < 4:
            return None
        sales  = q['매출액(vat 제외)'].values
        prev   = sales[-4:-2].mean()
        recent = sales[-2:].mean()
        if prev == 0:
            return None
        return (recent - prev) / prev

    def top3_products(hosp_name):
        d   = df_d[df_d['거래처명']==hosp_name]
        top = d.groupby('품명요약2')['매출수량'].sum().sort_values(ascending=False).head(3)
        return ' / '.join([f"{prod} {int(qty)}개" for prod, qty in top.items()])

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
    features['거래기간_년']  = features['활동기간_일'] / 365
    features['신규여부']     = (
        (features['첫구매일'] >= ref_date - pd.DateOffset(months=6)) &
        (features['총구매횟수'] <= 4)
    )

    with st.spinner("거래처 패턴 분석 중..."):
        features['성장률']   = features['거래처명'].apply(get_growth)
        features['주요제품'] = features['거래처명'].apply(top3_products)

    def assign_group(row):
        inactive   = row['미구매일수']
        cnt        = row['총구매횟수']
        growth     = row['성장률']
        revenue    = row['누적매출액']
        is_new     = row['신규여부']
        avg_cycle  = max(row['평균구매주기'], 1)
        threshold  = avg_cycle * 1.5
        years      = row['거래기간_년']
        on_track   = inactive < threshold

        # 신규
        if is_new:
            return '🆕 신규'
        # 저효율
        if cnt <= 2 or inactive > 365:
            return '📉 저효율형'
        # 성장형
        if growth is not None and growth >= 0.1 and on_track:
            return '🚀 성장형'
        # 안정형
        if years >= 1 and cnt >= 6 and revenue >= 2_100_000 and on_track:
            return '✅ 안정형'
        # 위험형
        if cnt >= 3 and not on_track:
            return '⚠️ 위험형'
        return '⚠️ 위험형'

    features['그룹'] = features.apply(assign_group, axis=1)

    def normalize(series):
        mn, mx = series.min(), series.max()
        if mx == mn:
            return pd.Series([50.0]*len(series), index=series.index)
        return (series - mn) / (mx - mn) * 100

    growth_filled = features['성장률'].fillna(0)
    features['종합점수'] = (
        normalize(growth_filled)           * 0.35 +
        normalize(features['누적매출액'])  * 0.25 +
        normalize(features['총구매횟수'])  * 0.20 +
        normalize(features['구매제품수'])  * 0.10 +
        normalize(-features['미구매일수']) * 0.10
    ).round(1)

    # ── 요약 지표 ──────────────────────────────────────
    cols = st.columns(5)
    for col, label in zip(cols,
        ['🚀 성장형','✅ 안정형','⚠️ 위험형','📉 저효율형','🆕 신규']):
        col.metric(label, f"{(features['그룹']==label).sum()}개")

    st.divider()

    tab1, tab2, tab3 = st.tabs(["📋 거래처 목록","🔍 그룹별 패턴","📊 담당자 현황"])

    with tab1:
        col_f1, col_f2 = st.columns(2)
        mgr_list   = ['전체'] + sorted(features['담당자'].dropna().unique().tolist())
        group_list = ['전체','🚀 성장형','✅ 안정형','⚠️ 위험형','📉 저효율형','🆕 신규']
        selected_mgr   = col_f1.selectbox("담당자", mgr_list)
        selected_group = col_f2.selectbox("그룹",   group_list)

        # features는 항상 숫자 상태 유지
        # 필터 + 정렬을 숫자 상태에서 처리
        filt = features.copy()
        if selected_mgr   != '전체':
            filt = filt[filt['담당자'] == selected_mgr]
        if selected_group != '전체':
            filt = filt[filt['그룹']   == selected_group]

        filt = filt.sort_values('누적매출액', ascending=False)

        # 표시용 컬럼만 따로 만들기
        display = pd.DataFrame()
        display['거래처명']    = filt['거래처명'].values
        display['담당자']      = filt['담당자'].values
        display['지역']        = filt['지역'].values
        display['그룹']        = filt['그룹'].values
        display['종합점수']    = filt['종합점수'].values
        display['총구매횟수']  = filt['총구매횟수'].values
        display['구매제품수']  = filt['구매제품수'].values
        display['누적매출액']  = filt['누적매출액'].apply(lambda x: f"{x:,.0f}원").values
        display['성장률']      = filt['성장률'].apply(
            lambda x: f"+{x:.0%}" if x is not None and x > 0
            else (f"{x:.0%}" if x is not None else "N/A")
        ).values
        display['미구매일수']   = filt['미구매일수'].values
        display['평균구매주기'] = filt['평균구매주기'].apply(lambda x: f"{x:.0f}일").values
        display['주요제품']     = filt['주요제품'].values

        st.dataframe(display, use_container_width=True, hide_index=True)

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
                 '누적매출액','미구매일수','평균구매주기']]
            .describe().round(1),
            use_container_width=True
        )

    with tab3:
        mgr_summary = features.groupby('담당자').agg(
            담당거래처=('거래처명',  'count'),
            성장형=('그룹',   lambda x: (x=='🚀 성장형').sum()),
            안정형=('그룹',   lambda x: (x=='✅ 안정형').sum()),
            위험형=('그룹',   lambda x: (x=='⚠️ 위험형').sum()),
            저효율형=('그룹', lambda x: (x=='📉 저효율형').sum()),
            신규=('그룹',     lambda x: (x=='🆕 신규').sum()),
            평균종합점수=('종합점수', 'mean'),
            평균매출=('누적매출액',  'mean'),
        ).reset_index()
        mgr_summary['평균종합점수'] = mgr_summary['평균종합점수'].round(1)
        mgr_summary['평균매출']    = mgr_summary['평균매출'].apply(
            lambda x: f"{x:,.0f}원"
        )
        st.dataframe(mgr_summary, use_container_width=True, hide_index=True)
