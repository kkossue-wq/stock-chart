"""주식 차트 분석기 - 메인 진입점"""

import os

import streamlit as st

# Streamlit Cloud 배포 시: secrets → 환경변수 자동 주입 (로컬은 .env 파일 사용)
try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, str):
            os.environ.setdefault(_k, _v)
except Exception:
    pass

from pages import (
    page_chart_analysis,
    page_market,
    page_portfolio,
    page_simulator,
    page_trade_journal,
    page_watchlist,
)

st.set_page_config(
    page_title="주식 차트 분석기",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = {
    "💼 내 포트폴리오": page_portfolio,
    "📈 차트 분석": page_chart_analysis,
    "⭐ 관심종목": page_watchlist,
    "🧮 매수 시뮬레이터": page_simulator,
    "📓 매매 일지": page_trade_journal,
    "🌡️ 시장 지표": page_market,
}


def main():
    with st.sidebar:
        st.markdown("## 📊 주식차트보는놈")

        # 포트폴리오 → 차트분석 자동 이동
        if st.session_state.get("nav_to"):
            st.session_state["_nav_radio"] = st.session_state.pop("nav_to")

        page_name = st.radio(
            "페이지",
            list(PAGES.keys()),
            key="_nav_radio",
            label_visibility="collapsed",
        )
        st.divider()

    PAGES[page_name]()

    with st.sidebar:
        st.divider()
        with st.expander("📌 API 키 설정"):
            st.markdown("""
**Claude AI 분석:**
로컬: `.env` 파일 / 클라우드: Streamlit Secrets에 `ANTHROPIC_API_KEY` 입력

**한투 계좌 연동:**
로컬: `.env` 파일 / 클라우드: Streamlit Secrets에 `KIS_APP_KEY`, `KIS_APP_SECRET`, `KIS_ACCOUNT_NO` 입력

→ [한투 API 발급](https://apiportal.koreainvestment.com)
            """)
        st.caption("⚠️ 투자 참고용 도구입니다. 투자 책임은 본인에게 있습니다.")


if __name__ == "__main__":
    main()
