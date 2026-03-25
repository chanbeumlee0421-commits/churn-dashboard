import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="거래처 성장 대시보드", layout="wide")
st.title("🐾 거래처 성장 대시보드")
st.caption("경보제약 동물의약품 | 기준일: 2026-03-24")

st.sidebar.markdown("""
### 그룹 분류 기준

| 그룹 | 기준 |
|---|---|
| 🟢 안심 | 누적 1000만↑ + 10회↑ + 주기배율 2.0↓ / 누적 3000만↑ + 3회↑ + 주기배율 2.0↓ |
| 🚀 성장 | 주기배율 1.5↓ + 반기추세 양수 + 이전반기 총매출 10%↑ + 3회↑ |
| 🌱 가능성 | 주기배율 1.5↓ + 최근 반기 활성 + 이전반기 거의 없었던 곳 |
| ⚠️ 주의 | 주기배율 1.5↑ + 반기추세 -30%↓ (과거엔 좋았지만 최근 신호 나쁨) |
| 😐 보통 | 위 조건 해당 없음 |
| 💀 정리 | 365일↑ 미구매 + 누적 300만↓ + 3회↓ |

---
### 지표 설명
**주기배율** = 미구매일수 ÷ 평균구매주기
**반기추세** = 최근6개월 vs 이전6개월 매출 변화율
**품목확장** = 최근 구매 제품수 - 초기 구매 제품수
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
    cut6  = ref_date - pd.DateOffset(months=6)
    cut12 = ref_date - pd.DateOffset(months=12)

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

    # ── 반기 매출 계산 ─────────────────────────────────
    def get_half_sales(hosp):
        d      = df_d[df_d['거래처명'] == hosp]
        recent = d[d['매출일(배송완료일)'] >  cut6]['매출액(vat 제외)'].sum()
        prev   = d[(d['매출일(배송완료일)'] >  cut12) &
                   (d['매출일(배송완료일)'] <= cut6)]['매출액(vat 제외)'].sum()
        return recent, prev

    # ── 품목 확장 ──────────────────────────────────────
    def get_expansion(hosp):
        d = df_d[df_d['거래처명'] == hosp].sort_values('매출일(배송완료일)')
        dates = d['매출일(배송완료일)'].unique()
        if len(dates) < 2:
            return 0
        early = d[d['매출일(배송완료일)'].isin(dates[:2])]['품명요약2'].nunique()
        late  = d[d['매출일(배송완료일)'].isin(dates[-2:])]['품명요약2'].nunique()
        return late - early

    # ── 주요제품 ──────────────────────────────────────
    def top3_products(hosp):
        d   = df_d[df_d['거래처명'] == hosp]
        top = d.groupby('품명요약2')['매출수량'].sum().sort_values(ascending=False).head(3)
        return ' / '.join([f"{p} {int(q)}개" for p, q in top.items()])

    with st.spinner("거래처 분석 중..."):
        half = features['거래처명'].apply(lambda x: pd.Series(get_half_sales(x), index=['최근반기','이전반기']))
        features['최근반기'] = half['최근반기'].values
        features['이전반기'] = half['이전반기'].values
        features['반기추세'] = features.apply(
            lambda r: (r['최근반기'] - r['이전반기']) / r['이전반기']
            if r['이전반기'] > 0 else None, axis=1)
        features['품목확장'] = features['거래처명'].apply(get_expansion)
        features['주요제품'] = features['거래처명'].apply(top3_products)

    # ── 그룹 분류 ──────────────────────────────────────
    def assign_group(row):
        cnt      = row['총구매횟수']
        ratio    = row['주기배율']
        revenue  = row['누적매출액']
        trend    = row['반기추세']
        recent6  = row['최근반기']
        prev6    = row['이전반기']
        total    = revenue if revenue > 0 else 1

        on_track = ratio < 1.5

        # 💀 정리: 365일↑ 미구매 + 누적 300만↓ + 3회↓
        if row['미구매일수'] >= 365 and revenue < 3_000_000 and cnt <= 3:
            return '💀 정리대상'

        # 🟢 안심
        if (revenue >= 10_000_000 and cnt >= 10 and ratio < 2.0):
            return '🟢 안심'
        if (revenue >= 30_000_000 and cnt >= 3 and ratio < 2.0):
            return '🟢 안심'

        # ⚠️ 주의: 과거엔 좋았는데 최근 신호 나쁨
        if (revenue >= 5_000_000 and cnt >= 5 and
            ratio >= 1.5 and
            trend is not None and trend <= -0.3):
            return '⚠️ 주의'

        # 🚀 성장: 진짜 성장 중
        # 이전반기가 총매출의 10% 이상 = 이전에도 거래 있었음
        if (on_track and cnt >= 3 and
            trend is not None and trend > 0 and
            prev6 >= total * 0.1):
            return '🚀 성장'

        # 🌱 가능성: 최근 활성화 or 재활성화
        # 최근 반기에 거래 있고, 이전반기 거의 없었던 곳
        if (on_track and cnt >= 2 and recent6 > 0 and
            (prev6 < total * 0.1 or trend is None)):
            return '🌱 가능성'

        # 😐 보통
        if on_track:
            return '😐 보통'

        return '😐 보통'

    features['그룹'] = features.apply(assign_group, axis=1)

    # ── 전체 현황 ──────────────────────────────────────
    st.subheader("📊 전체 현황")
    total  = len(features)
    groups = ['🟢 안심', '🚀 성장', '🌱 가능성', '⚠️ 주의', '😐 보통', '💀 정리대상']
    counts = {g: (features['그룹'] == g).sum() for g in groups}

    cols = st.columns(6)
    for col, (label, cnt) in zip(cols, counts.items()):
        col.metric(label, f"{cnt}개", f"{cnt/total:.0%}")

    color_map = {
        '🟢 안심':    '#2ecc71',
        '🚀 성장':    '#3498db',
        '🌱 가능성':  '#f1c40f',
        '⚠️ 주의':   '#e67e22',
        '😐 보통':    '#95a5a6',
        '💀 정리대상':'#e74c3c',
    }
    pie = pd.DataFrame({'그룹': list(counts.keys()), '수': list(counts.values())})
    fig = px.pie(pie, values='수', names='그룹',
                 color='그룹', color_discrete_map=color_map)
    fig.update_layout(height=300, margin=dict(t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── 필터 + 테이블 ──────────────────────────────────
    col_f1, col_f2 = st.columns(2)
    mgr_list   = ['전체'] + sorted(features['담당자'].dropna().unique().tolist())
    group_list = ['전체'] + groups
    selected_mgr   = col_f1.selectbox("담당자", mgr_list)
    selected_group = col_f2.selectbox("그룹",   group_list)

    result = features.copy()
    if selected_mgr   != '전체':
        result = result[result['담당자'] == selected_mgr]
    if selected_group != '전체':
        result = result[result['그룹']   == selected_group]

    result = result.sort_values('누적매출액', ascending=False)

    display = pd.DataFrame()
    display['거래처명']    = result['거래처명'].values
    display['담당자']      = result['담당자'].values
    display['지역']        = result['지역'].values
    display['그룹']        = result['그룹'].values
    display['총구매횟수']  = result['총구매횟수'].values
    display['구매제품수']  = result['구매제품수'].values
    display['누적매출액']  = result['누적매출액'].apply(lambda x: f"{x:,.0f}원").values
    display['회당매출']    = result['회당매출'].apply(lambda x: f"{x:,.0f}원").values
    display['반기추세']    = result['반기추세'].apply(
        lambda x: f"+{x:.0%}" if x is not None and x > 0
        else (f"{x:.0%}" if x is not None else "N/A")).values
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
        안심=('그룹',    lambda x: (x == '🟢 안심').sum()),
        성장=('그룹',    lambda x: (x == '🚀 성장').sum()),
        가능성=('그룹',  lambda x: (x == '🌱 가능성').sum()),
        주의=('그룹',    lambda x: (x == '⚠️ 주의').sum()),
        보통=('그룹',    lambda x: (x == '😐 보통').sum()),
        정리=('그룹',    lambda x: (x == '💀 정리대상').sum()),
        평균매출=('누적매출액', 'mean'),
    ).reset_index()
    mgr_sum['평균매출'] = mgr_sum['평균매출'].apply(lambda x: f"{x:,.0f}원")
    st.dataframe(mgr_sum, use_container_width=True, hide_index=True)
