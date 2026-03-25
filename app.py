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
| 🚀 성장형 | 분기 매출이 우상향 중 + 최근 180일 내 구매 |
| ✅ 안정형 | 꾸준한 구매 + 매출 유지 |
| ⚠️ 위험형 | 매출 감소 중 or 구매 간격 늘어남 |
| 📉 저효율형 | 365일 이상 미구매 or 1회 구매 후 없음 |

---
### 종합점수 계산 방식 (100점 만점)
| 항목 | 가중치 | 설명 |
|---|---|---|
| 분기성장률 | 30% | 분기별 매출이 얼마나 오르는지 |
| 누적매출액 | 25% | 전체 거래처 대비 상대적 매출 크기 |
| 구매횟수 | 20% | 얼마나 자주 사는지 |
| 제품다양성 | 15% | 몇 가지 제품을 사는지 |
| 최근성 | 10% | 마지막 구매가 얼마나 최근인지 |

---
### 분기 매출트렌드 설명
- 거래 기간을 분기(3개월)로 나눠
  각 분기 매출의 기울기(증가/감소 방향)
- 양수(+): 분기별로 매출 증가 중
- 음수(-): 분기별로 매출 감소 중
- 거래 2분기 미만은 0으로 표시
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
    df_d['분기'] = df_d['매출일(배송완료일)'].dt.to_period('Q')
    ref_date = pd.Timestamp('2026-03-24')

    # ── 분기별 매출 트렌드 계산 ────────────────────────
    def get_quarterly_trend(hosp_name):
        d = df_d[df_d['거래처명'] == hosp_name]
        qtr = d.groupby('분기')['매출액(vat 제외)'].sum().sort_index()
        if len(qtr) < 2:
            return 0.0
        # 선형 회귀 기울기 (분기 순서 vs 매출)
        x = np.arange(len(qtr))
        y = qtr.values
        if y.std() == 0:
            return 0.0
        slope = np.polyfit(x, y, 1)[0]
        # 평균 매출 대비 기울기 비율
        return slope / y.mean() if y.mean() != 0 else 0.0

    # ── 주요제품 상위 3개 + 수량 ───────────────────────
    def top3_products(hosp_name):
        d = df_d[df_d['거래처명'] == hosp_name]
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

    with st.spinner("거래처 패턴 분석 중... (잠시 기다려주세요)"):
        features['분기성장률'] = features['거래처명'].apply(get_quarterly_trend)
        features['주요제품']   = features['거래처명'].apply(top3_products)

    # ── 종합점수 계산 ──────────────────────────────────
    def normalize(series):
        mn, mx = series.min(), series.max()
        if mx == mn:
            return pd.Series([50.0] * len(series), index=series.index)
        return (series - mn) / (mx - mn) * 100

    features['종합점수'] = (
        normalize(features['분기성장률'])  * 0.30 +
        normalize(features['누적매출액'])  * 0.25 +
        normalize(features['총구매횟수'])  * 0.20 +
        normalize(features['구매제품수'])  * 0.15 +
        normalize(-features['미구매일수']) * 0.10
    )

    # ── 그룹 분류 ──────────────────────────────────────
    def assign_group(row):
        # 저효율: 1년 이상 미구매 or 1회만 구매
        if row['미구매일수'] > 365 or row['총구매횟수'] <= 1:
            return '📉 저효율형'
        # 성장형: 분기 매출 우상향 + 최근 180일 내 구매
        if row['분기성장률'] > 0 and row['미구매일수'] <= 180:
            return '🚀 성장형'
        # 위험형: 분기 매출 감소 or 구매 간격 길어짐
        if row['분기성장률'] < -0.1 or row['미구매일수'] > 180:
            return '⚠️ 위험형'
        # 안정형: 나머지
        return '✅ 안정형'

    features['그룹'] = features.apply(assign_group, axis=1)

    # ── 요약 지표 ──────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    for col, label in zip(
        [col1, col2, col3, col4],
        ['🚀 성장형', '✅ 안정형', '⚠️ 위험형', '📉 저효율형']
    ):
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

        result = features[[
            '거래처명', '담당자', '지역', '그룹', '종합점수',
            '총구매횟수', '구매제품수', '누적매출액',
            '분기성장률', '미구매일수', '주요제품'
        ]].copy()

        if selected_mgr   != '전체':
            result = result[result['담당자'] == selected_mgr]
        if selected_group != '전체':
            result = result[result['그룹']   == selected_group]

        result = result.sort_values('종합점수', ascending=False)
        result['종합점수']  = result['종합점수'].round(1)
        result['분기성장률'] = result['분기성장률'].apply(
            lambda x: f"+{x:.1%}" if x > 0 else f"{x:.1%}"
        )
        result['누적매출액'] = result['누적매출액'].apply(lambda x: f"{x:,.0f}원")

        st.dataframe(result, use_container_width=True, hide_index=True)

    with tab2:
        selected_g = st.selectbox(
            "분석할 그룹 선택",
            ['🚀 성장형', '✅ 안정형', '⚠️ 위험형', '📉 저효율형']
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
            .sort_values(ascending=False)
            .head(5)
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
            grp[['종합점수', '총구매횟수', '구매제품수',
                 '누적매출액', '분기성장률', '미구매일수']]
            .describe().round(1),
            use_container_width=True
        )

    with tab3:
        mgr_summary = features.groupby('담당자').agg(
            담당거래처=('거래처명',  'count'),
            성장형=('그룹', lambda x: (x=='🚀 성장형').sum()),
            안정형=('그룹', lambda x: (x=='✅ 안정형').sum()),
            위험형=('그룹', lambda x: (x=='⚠️ 위험형').sum()),
            저효율형=('그룹', lambda x: (x=='📉 저효율형').sum()),
            평균종합점수=('종합점수', 'mean'),
            평균매출=('누적매출액', 'mean'),
        ).reset_index()
        mgr_summary['평균종합점수'] = mgr_summary['평균종합점수'].round(1)
        mgr_summary['평균매출']    = mgr_summary['평균매출'].apply(
            lambda x: f"{x:,.0f}원"
        )
        st.dataframe(mgr_summary, use_container_width=True, hide_index=True)
