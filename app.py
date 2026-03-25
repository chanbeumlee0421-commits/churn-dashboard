import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(page_title="거래처 분석 대시보드", layout="wide")
st.title("🐾 거래처 성장 분석 대시보드")
st.caption("경보제약 동물의약품 | 기준일: 2026-03-24")

st.sidebar.markdown("""
### 그룹 분류 기준
| 그룹 | 기준 |
|---|---|
| 🚀 성장형 | 매출 증가 + 최근 구매 + 다품목 |
| ✅ 안정형 | 매출 유지 + 정기 구매 |
| ⚠️ 위험형 | 매출 감소 또는 구매 간격 늘어남 |
| 📉 저효율형 | 장기 미구매 + 저매출 |

**매출트렌드 설명**
- 전체 거래 기간을 전반/후반으로 나눠
  후반 매출이 전반 대비 얼마나 변했는지
- +50% = 후반 매출이 전반보다 50% 증가
- -30% = 후반 매출이 전반보다 30% 감소
- 거래 3회 미만은 계산 제외

**주요제품**
- 구매 횟수 기준 상위 3개 제품
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
    features['평균구매주기'] = (features['활동기간_일'] / features['총구매횟수'].replace(0,1))
    features['거래처당매출'] = (features['누적매출액'] / features['총구매횟수'].replace(0,1))

    # 주요제품 상위 3개
    def top3_products(hosp):
        prods = df_d[df_d['거래처명']==hosp]['품명요약2'].value_counts().head(3)
        return ', '.join(prods.index.tolist())

    # 매출 트렌드 (3회 이상만 계산)
    def get_trend(row):
        if row['총구매횟수'] < 3:
            return 0.0
        hosp = row['거래처명']
        mid  = row['첫구매일'] + (row['마지막구매일'] - row['첫구매일']) / 2
        d    = df_d[df_d['거래처명'] == hosp]
        early = d[d['매출일(배송완료일)'] <= mid]['매출액(vat 제외)'].sum()
        late  = d[d['매출일(배송완료일)'] >  mid]['매출액(vat 제외)'].sum()
        if early == 0: return 0.0
        return (late - early) / early

    with st.spinner("거래처 패턴 분석 중..."):
        features['매출트렌드'] = features.apply(get_trend, axis=1)
        features['주요제품']   = features['거래처명'].apply(top3_products)

    # ── 그룹 점수화 (설득력 있는 기준) ────────────────
    # 각 지표를 0~100점으로 정규화 후 가중 합산
    def normalize(series):
        mn, mx = series.min(), series.max()
        if mx == mn: return pd.Series([50]*len(series), index=series.index)
        return (series - mn) / (mx - mn) * 100

    # 점수 높을수록 좋음
    features['점수_매출액']    = normalize(features['누적매출액'])
    features['점수_구매횟수']  = normalize(features['총구매횟수'])
    features['점수_제품다양성'] = normalize(features['구매제품수'])
    features['점수_트렌드']    = normalize(features['매출트렌드'])
    features['점수_최근성']    = normalize(-features['미구매일수'])  # 최근일수록 높음
    features['점수_구매주기']  = normalize(-features['평균구매주기']) # 주기 짧을수록 높음

    # 가중 합산 (총합 100%)
    features['종합점수'] = (
        features['점수_매출액']    * 0.25 +
        features['점수_구매횟수']  * 0.20 +
        features['점수_제품다양성'] * 0.15 +
        features['점수_트렌드']    * 0.20 +
        features['점수_최근성']    * 0.15 +
        features['점수_구매주기']  * 0.05
    )

    # 그룹 분류 (종합점수 + 트렌드 조합)
    def assign_group(row):
        score = row['종합점수']
        trend = row['매출트렌드']
        inactive = row['미구매일수']

        if inactive > 365:
            return '📉 저효율형'
        elif score >= 60 and trend >= 0:
            return '🚀 성장형'
        elif score >= 40 and inactive <= 180:
            return '✅ 안정형'
        elif trend < -0.2 or inactive > 180:
            return '⚠️ 위험형'
        else:
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
            '매출트렌드', '미구매일수', '주요제품'
        ]].copy()

        if selected_mgr   != '전체':
            result = result[result['담당자'] == selected_mgr]
        if selected_group != '전체':
            result = result[result['그룹']   == selected_group]

        result = result.sort_values('종합점수', ascending=False)
        result['종합점수']  = result['종합점수'].round(1)
        result['매출트렌드'] = result['매출트렌드'].apply(
            lambda x: f"+{x:.0%}" if x > 0 else f"{x:.0%}"
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

        st.markdown("**📦 주요 구매 제품 Top 5**")
        prod_cnt = df_d[df_d['거래처명'].isin(grp['거래처명'])]['품명요약2'].value_counts().head(5)
        st.bar_chart(prod_cnt)

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
                 '누적매출액', '매출트렌드', '미구매일수']].describe().round(1),
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
        mgr_summary['평균매출']    = mgr_summary['평균매출'].apply(lambda x: f"{x:,.0f}원")
        st.dataframe(mgr_summary, use_container_width=True, hide_index=True)
