import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="거래처 성장 대시보드", layout="wide")
st.title("🐾 거래처 성장 대시보드")
st.caption("경보제약 동물의약품 | 기준일: 2026-03-24")

st.sidebar.markdown("""
### 그룹 분류 기준

| 그룹 | 의미 |
|---|---|
| 🟢 안심 | 궤도 안착, 안정적 유지 |
| 🚀 성장 | 매출·품목 모두 성장 중 |
| 🌱 가능성 | 아직 작지만 방향이 좋음 |
| 😐 보통 | 현상 유지, 성장 없음 |
| 💀 정리 | 재활성화 가능성 낮음 |

---
### 측정 기준
**회당매출**: 누적매출 ÷ 구매횟수
**평균주기**: 활동기간 ÷ 구매횟수
**주기배율**: 미구매일수 ÷ 평균주기
**품목확장**: 최근 구매 제품수 - 초기 구매 제품수
**매출추세**: 최근 2회 평균 - 초기 2회 평균
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
    features['평균구매주기'] = features['활동기간_일'] / features['총구매횟수'].replace(0, 1)
    features['주기배율']     = features['미구매일수'] / features['평균구매주기'].replace(0, 1)
    features['회당매출']     = features['누적매출액'] / features['총구매횟수'].replace(0, 1)

    # ── 품목 확장 ──────────────────────────────────────
    def get_expansion(hosp):
        d = df_d[df_d['거래처명'] == hosp].sort_values('매출일(배송완료일)')
        dates = d['매출일(배송완료일)'].unique()
        if len(dates) < 2:
            return 0
        early = d[d['매출일(배송완료일)'].isin(dates[:2])]['품명요약2'].nunique()
        late  = d[d['매출일(배송완료일)'].isin(dates[-2:])]['품명요약2'].nunique()
        return late - early

    # ── 매출 추세 ──────────────────────────────────────
    def get_sales_trend(hosp, cnt):
        if cnt < 2:
            return 0
        d = df_d[df_d['거래처명'] == hosp]
        cut = ref_date - pd.DateOffset(months=6)
        recent = d[d['매출일(배송완료일)'] >  cut]['매출액(vat 제외)'].sum()
        prev   = d[d['매출일(배송완료일)'] <= cut]['매출액(vat 제외)'].sum()
        if prev == 0:
            return 0  # 이전 반기 없으면 신규로 보고 0
        return (recent - prev) / prev

    # ── 주요제품 ──────────────────────────────────────
    def top3_products(hosp):
        d   = df_d[df_d['거래처명'] == hosp]
        top = d.groupby('품명요약2')['매출수량'].sum().sort_values(ascending=False).head(3)
        return ' / '.join([f"{p} {int(q)}개" for p, q in top.items()])

    with st.spinner("거래처 분석 중..."):
        features['품목확장']  = features['거래처명'].apply(get_expansion)
        features['매출추세']  = features.apply(
            lambda r: get_sales_trend(r['거래처명'], r['총구매횟수']), axis=1)
        features['주요제품']  = features['거래처명'].apply(top3_products)

    # ── 그룹 분류 ──────────────────────────────────────
    def assign_group(row):
        cnt       = row['총구매횟수']
        inactive  = row['미구매일수']
        ratio     = row['주기배율']
        revenue   = row['누적매출액']
        per_rev   = row['회당매출']
        expansion = row['품목확장']
        trend     = row['매출추세']
        prod_cnt  = row['구매제품수']

        # 주기 정상 여부 (회당매출 크면 주기 여유 줌)
        if per_rev >= 1_000_000:
            on_track = ratio < 2.0   # 대량구매는 2배까지 허용
        else:
            on_track = ratio < 1.5

        # 💀 정리: 주기 2배 초과 + 회당매출 100만원 미만 + 3회 이하
        if ratio > 2.0 and per_rev < 1_000_000 and cnt <= 3:
            return '💀 정리대상'

        # 판단 불가: 1회 구매
        if cnt == 1:
            if inactive > 270:
                return '💀 정리대상'
            return '🌱 가능성'

        # 🟢 안심: 꾸준하고 안정적
        if on_track and (
            cnt >= 5 or
            (cnt >= 3 and per_rev >= 1_000_000)
        ) and (prod_cnt >= 2 or per_rev >= 1_000_000):
            return '🟢 안심'

        # 🚀 성장: 매출 + 품목 모두 증가
        if on_track and cnt >= 3 and trend > 0.1 and expansion >= 0:
            return '🚀 성장'

        # 🌱 가능성: 아직 작지만 방향 좋음
        if on_track and cnt >= 2 and (trend > 0 or expansion > 0):
            return '🌱 가능성'

        # 😐 보통: 오긴 오는데 성장 없음
        if on_track:
            return '😐 보통'

        # 나머지
        return '💀 정리대상'

    features['그룹'] = features.apply(assign_group, axis=1)

    # ── 전체 현황 ──────────────────────────────────────
    st.subheader("📊 전체 현황")

    total  = len(features)
    counts = {g: (features['그룹']==g).sum()
              for g in ['🟢 안심','🚀 성장','🌱 가능성','😐 보통','💀 정리대상']}

    c1,c2,c3,c4,c5 = st.columns(5)
    for col, (label, cnt) in zip([c1,c2,c3,c4,c5], counts.items()):
        col.metric(label, f"{cnt}개", f"{cnt/total:.0%}")

    pie = pd.DataFrame({'그룹': list(counts.keys()), '수': list(counts.values())})
    color_map = {
        '🟢 안심':    '#2ecc71',
        '🚀 성장':    '#3498db',
        '🌱 가능성':  '#f1c40f',
        '😐 보통':    '#95a5a6',
        '💀 정리대상':'#e74c3c'
    }
    fig = px.pie(pie, values='수', names='그룹',
                 color='그룹', color_discrete_map=color_map)
    fig.update_layout(height=300, margin=dict(t=0,b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── 필터 + 테이블 ──────────────────────────────────
    col_f1, col_f2, col_f3 = st.columns(3)
    mgr_list    = ['전체'] + sorted(features['담당자'].dropna().unique().tolist())
    group_list  = ['전체','🟢 안심','🚀 성장','🌱 가능성','😐 보통','💀 정리대상']
    selected_mgr   = col_f1.selectbox("담당자", mgr_list)
    selected_group = col_f2.selectbox("그룹",   group_list)

    # 숫자 상태에서 필터 + 정렬
    result = features.copy()
    if selected_mgr   != '전체':
        result = result[result['담당자'] == selected_mgr]
    if selected_group != '전체':
        result = result[result['그룹']   == selected_group]

    result = result.sort_values('누적매출액', ascending=False)

    # 표시용 변환
    display = pd.DataFrame()
    display['거래처명']    = result['거래처명'].values
    display['담당자']      = result['담당자'].values
    display['지역']        = result['지역'].values
    display['그룹']        = result['그룹'].values
    display['총구매횟수']  = result['총구매횟수'].values
    display['구매제품수']  = result['구매제품수'].values
    display['누적매출액']  = result['누적매출액'].apply(lambda x: f"{x:,.0f}원").values
    display['회당매출']    = result['회당매출'].apply(lambda x: f"{x:,.0f}원").values
    display['매출추세']    = result['매출추세'].apply(
        lambda x: f"+{x:.0%}" if x > 0 else f"{x:.0%}").values
    display['품목확장']    = result['품목확장'].apply(
        lambda x: f"+{int(x)}" if x > 0 else str(int(x))).values
    display['미구매일수']  = result['미구매일수'].values
    display['평균구매주기']= result['평균구매주기'].apply(lambda x: f"{x:.0f}일").values
    display['주기배율']    = result['주기배율'].apply(lambda x: f"{x:.1f}배").values
    display['주요제품']    = result['주요제품'].values

    st.subheader(f"📋 거래처 목록 ({len(result)}개)")
    st.dataframe(display, use_container_width=True, hide_index=True)

    st.divider()

    # ── 담당자 현황 ────────────────────────────────────
    st.subheader("👤 담당자별 현황")
    mgr_sum = features.groupby('담당자').agg(
        담당거래처=('거래처명',  'count'),
        안심=('그룹', lambda x: (x=='🟢 안심').sum()),
        성장=('그룹', lambda x: (x=='🚀 성장').sum()),
        가능성=('그룹', lambda x: (x=='🌱 가능성').sum()),
        보통=('그룹', lambda x: (x=='😐 보통').sum()),
        정리=('그룹', lambda x: (x=='💀 정리대상').sum()),
        평균매출=('누적매출액', 'mean'),
    ).reset_index()
    mgr_sum['평균매출'] = mgr_sum['평균매출'].apply(lambda x: f"{x:,.0f}원")
    st.dataframe(mgr_sum, use_container_width=True, hide_index=True)
