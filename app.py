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
| 🟢 안심 | 누적 1000만↑+10회↑+주기배율 2.0↓ / 누적 3000만↑+3회↑+주기배율 2.0↓ |
| 🚀 성장 | 반기추세 +20%↑(이전반기 50만↑) / 급성장 / 장기재활성화 |
| ⚠️ 주의 | 누적 500만↑+5회↑+주기배율 1.5↑+반기추세 -30%↓ |
| 😐 보통 | 위 조건 해당 없음 |
| 💀 정리 | 미구매 365일↑+누적 300만↓+3회↓ |

---
### 지표 설명
**주기배율** = 미구매일수 ÷ 평균구매주기
**반기추세** = 최근6개월 vs 이전6개월 매출 변화율
(이전반기 50만원 미만이면 계산 제외 → -)
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

    # ── 반기 매출 ──────────────────────────────────────
    def get_half(hosp):
        d      = df_d[df_d['거래처명'] == hosp]
        recent = d[d['매출일(배송완료일)'] >  cut6]['매출액(vat 제외)'].sum()
        prev   = d[(d['매출일(배송완료일)'] >  cut12) &
                   (d['매출일(배송완료일)'] <= cut6)]['매출액(vat 제외)'].sum()
        return recent, prev

    # ── 주요제품 ──────────────────────────────────────
    def top3_products(hosp):
        d   = df_d[df_d['거래처명'] == hosp]
        top = d.groupby('품명요약2')['매출수량'].sum().sort_values(ascending=False).head(3)
        return ' / '.join([f"{p} {int(q)}개" for p, q in top.items()])

    with st.spinner("거래처 분석 중..."):
        half = features['거래처명'].apply(
            lambda x: pd.Series(get_half(x), index=['최근반기', '이전반기']))
        features['최근반기'] = half['최근반기'].values
        features['이전반기'] = half['이전반기'].values

        # 이전반기 50만원 미만이면 None (왜곡 방지)
        features['반기추세'] = features.apply(
            lambda r: (r['최근반기'] - r['이전반기']) / r['이전반기']
            if r['이전반기'] >= 500_000 else None, axis=1)

        features['주요제품'] = features['거래처명'].apply(top3_products)

    # ── 그룹 분류 ──────────────────────────────────────
    def assign_group(row):
        cnt      = row['총구매횟수']
        ratio    = row['주기배율']
        revenue  = row['누적매출액']
        trend    = row['반기추세']
        recent6  = row['최근반기']
        prev6    = row['이전반기']
        inactive = row['미구매일수']
        duration = row['활동기간_일']
        on_track = ratio < 1.5

        # 💀 정리
        if inactive >= 365 and revenue < 3_000_000 and cnt <= 3:
            return '💀 정리대상'

        # 🟢 안심
        if revenue >= 10_000_000 and cnt >= 10 and ratio < 2.0:
            return '🟢 안심'
        if revenue >= 30_000_000 and cnt >= 3 and ratio < 2.0:
            return '🟢 안심'

        # ⚠️ 주의
        if (revenue >= 5_000_000 and cnt >= 5 and
                ratio >= 1.5 and
                trend is not None and trend <= -0.3):
            return '⚠️ 주의'

        # 🚀 성장 ① 반기추세 +20%↑ (이전반기 50만↑ 보장됨)
        if on_track and trend is not None and trend >= 0.2:
            return '🚀 성장'

        # 🚀 성장 ② 급성장 (이전반기 50만↑ + 최근반기 3배↑ + 활동 180일↑)
        if (on_track and prev6 >= 500_000 and
                recent6 >= prev6 * 3 and duration >= 180):
            return '🚀 성장'

        # 🚀 성장 ③ 장기재활성화 (활동 365일↑ + 이전반기 0 + 최근반기 500만↑)
        if (on_track and duration >= 365 and
                prev6 == 0 and recent6 >= 5_000_000):
            return '🚀 성장'

        # 😐 보통
        return '😐 보통'

    features['그룹'] = features.apply(assign_group, axis=1)

    # ── 전체 현황 ──────────────────────────────────────
    st.subheader("📊 전체 현황")
    total  = len(features)
    groups = ['🟢 안심', '🚀 성장', '⚠️ 주의', '😐 보통', '💀 정리대상']
    counts = {g: (features['그룹'] == g).sum() for g in groups}

    cols = st.columns(5)
    for col, (label, cnt) in zip(cols, counts.items()):
        col.metric(label, f"{cnt}개", f"{cnt/total:.0%}")

    color_map = {
        '🟢 안심':    '#2ecc71',
        '🚀 성장':    '#3498db',
        '⚠️ 주의':   '#e67e22',
        '😐 보통':    '#95a5a6',
        '💀 정리대상':'#e74c3c',
    }
    pie = pd.DataFrame({'그룹': list(counts.
