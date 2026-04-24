"""
페이지 UI 모음
- 차트 분석 (메인)
- 관심종목 대시보드
- 매수 시뮬레이터
- 매매 일지 + AI 코칭
- 시장 지표 (공포탐욕/환율)
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from constants import (
    ALL_STOCKS,
    KOREAN_ETFS,
    KOREAN_STOCKS,
    US_ETFS,
    US_STOCKS,
    TICKER_TO_NAME,
    currency_name,
    format_price,
    is_korean,
)
from core import (
    add_to_watchlist,
    add_trade,
    ai_analyze_chart,
    ai_analyze_single_trade,
    ai_analyze_trades,
    analyze_signals,
    calculate_indicators,
    create_chart,
    delete_trade,
    extract_fundamentals,
    fear_greed_korean,
    find_support_resistance,
    get_dividends,
    get_exchange_rate,
    get_fear_greed,
    get_news,
    get_price_on_date,
    get_stock_data,
    load_trades,
    load_watchlist,
    remove_from_watchlist,
)


# =========================
# 공용: 종목 선택 UI
# =========================
def stock_picker(key_prefix: str = "") -> str:
    """사이드바/메인에서 쓸 수 있는 종목 선택 위젯. 티커 반환."""
    market = st.radio(
        "시장",
        ["한국 ETF 🇰🇷", "미국 ETF 🇺🇸", "한국 주식", "미국 주식", "직접 입력 ✏️"],
        key=f"{key_prefix}_market",
        horizontal=True,
    )
    ticker = ""
    if market == "한국 ETF 🇰🇷":
        key = st.selectbox("ETF 선택", list(KOREAN_ETFS.keys()), key=f"{key_prefix}_kr_etf")
        ticker = KOREAN_ETFS[key]
    elif market == "미국 ETF 🇺🇸":
        key = st.selectbox("ETF 선택", list(US_ETFS.keys()), key=f"{key_prefix}_us_etf")
        ticker = US_ETFS[key]
    elif market == "한국 주식":
        key = st.selectbox("종목 선택", list(KOREAN_STOCKS.keys()), key=f"{key_prefix}_kr")
        ticker = KOREAN_STOCKS[key]
    elif market == "미국 주식":
        key = st.selectbox("종목 선택", list(US_STOCKS.keys()), key=f"{key_prefix}_us")
        ticker = US_STOCKS[key]
    else:
        ticker = st.text_input(
            "티커 직접 입력",
            placeholder="360750.KS / SCHD",
            key=f"{key_prefix}_custom",
            help="한국 ETF 예: 360750.KS / 미국 ETF 예: SCHD, VOO",
        ).strip().upper()
    return ticker


def signal_icon(text: str) -> str:
    # 과매수/과매도 먼저 체크 (안에 '매수'/'매도' 포함되어 있어서 순서 중요)
    if "과매수" in text:
        return "🔴"
    if "과매도" in text:
        return "🟢"
    pos = ["매수", "골든", "반등", "긍정", "상승"]
    neg = ["매도", "데드", "조정", "부정", "하락"]
    if any(w in text for w in pos):
        return "🟢"
    if any(w in text for w in neg):
        return "🔴"
    return "🟡"


# =========================
# 1. 차트 분석 페이지
# =========================
def page_chart_analysis():
    st.title("📈 차트 분석")
    st.caption("한국 & 미국 주식 기술적 분석 + AI 조언")

    with st.sidebar:
        st.header("🔍 종목 선택")

        # 포트폴리오에서 넘어온 경우 자동 로드
        if "chart_ticker" in st.session_state:
            ticker = st.session_state.pop("chart_ticker")
            st.info(f"티커: `{ticker}`")
            if st.button("🔄 다른 종목 선택", use_container_width=True):
                st.rerun()
        else:
            ticker = stock_picker("chart")
            if ticker:
                st.caption(f"티커: `{ticker}`")

        st.divider()
        period = st.select_slider(
            "조회 기간",
            options=["3mo", "6mo", "1y", "2y", "5y"],
            value="1y",
            format_func=lambda x: {
                "3mo": "3개월", "6mo": "6개월",
                "1y": "1년", "2y": "2년", "5y": "5년",
            }[x],
        )
        st.divider()
        st.subheader("지표 표시")
        show_ma = st.checkbox("이동평균선 (MA20/60/120)", value=True)
        show_bb = st.checkbox("볼린저밴드", value=True)
        show_sr = st.checkbox("지지/저항선", value=True)
        st.divider()
        run_ai = st.button("🤖 AI 분석 실행", type="primary", use_container_width=True)

    if not ticker:
        _landing_help()
        return

    with st.spinner(f"📡 {ticker} 데이터 불러오는 중..."):
        df, info = get_stock_data(ticker, period)

    if df.empty:
        st.error(f"'{ticker}' 데이터를 불러올 수 없습니다. 티커 확인!")
        st.info("예: 한국 `005930.KS` / `086520.KQ` / 미국 `AAPL`")
        return

    df = calculate_indicators(df)
    signals = analyze_signals(df)
    support, resistance = find_support_resistance(df)
    news = get_news(ticker)
    fundamentals = extract_fundamentals(info)

    # 상단 메트릭
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest
    change_pct = (latest["Close"] - prev["Close"]) / prev["Close"] * 100

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        name = info.get("longName", ticker)
        st.metric(
            name[:20] + ("..." if len(name) > 20 else ""),
            format_price(latest["Close"], ticker),
            f"{change_pct:+.2f}%",
        )
    with m2:
        rsi_val = signals.get("rsi", {}).get("value")
        rsi_label = (
            "과매도" if rsi_val and rsi_val < 30
            else ("과매수" if rsi_val and rsi_val > 70 else "중립")
        )
        st.metric("RSI", f"{rsi_val}" if rsi_val else "N/A", rsi_label)
    with m3:
        overall = signals.get("overall", "중립")
        score = signals.get("score", 0)
        st.metric("종합 신호", overall, f"점수 {score:+d}")
    with m4:
        pe = fundamentals.get("PER (trailing)", "N/A")
        pb = fundamentals.get("PBR", "N/A")
        st.metric("PER / PBR", f"{pe} / {pb}")
    with m5:
        dy = fundamentals.get("배당수익률", "N/A")
        st.metric("배당수익률", dy)

    # 관심종목 추가 버튼
    col_a, col_b, _ = st.columns([1, 1, 3])
    watchlist = load_watchlist()
    already_in = any(w["ticker"] == ticker for w in watchlist)
    with col_a:
        if already_in:
            if st.button("⭐ 관심종목에서 제거", use_container_width=True):
                remove_from_watchlist(ticker)
                st.rerun()
        else:
            if st.button("☆ 관심종목 추가", use_container_width=True):
                add_to_watchlist(ticker, info.get("longName", ticker))
                st.success(f"{ticker} 관심종목에 추가!")
                st.rerun()

    # 차트
    st.plotly_chart(
        create_chart(df, show_bb, show_ma, show_sr, support, resistance),
        use_container_width=True,
    )

    # 신호 요약
    st.subheader("📊 기술적 신호 요약")
    c1, c2, c3, c4 = st.columns(4)
    for col, (name, key) in zip(
        [c1, c2, c3, c4],
        [("RSI", "rsi"), ("MACD", "macd"), ("볼린저밴드", "bollinger"), ("이동평균", "ma")],
    ):
        with col:
            sig = signals.get(key, {}).get("signal", "N/A")
            col.info(f"**{name}**\n\n{signal_icon(sig)} {sig}")

    # 탭
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["💡 매매 가이드", "📰 뉴스", "💰 배당/재무", "🎯 지지/저항", "⭐ 추천 ETF", "🤖 AI 분석"]
    )

    with tab1:
        _render_trade_guide(signals)
    with tab2:
        _render_news(news)
    with tab3:
        _render_fundamentals(fundamentals, ticker)
    with tab4:
        _render_support_resistance(support, resistance, latest["Close"], ticker)
    with tab5:
        _render_recommended_etfs()
    with tab6:
        if run_ai:
            with st.spinner("Claude AI가 분석 중입니다... (10~20초)"):
                analysis = ai_analyze_chart(
                    ticker, df, signals, info, news, fundamentals, support, resistance
                )
            st.markdown(analysis)
        else:
            st.info("사이드바의 **'🤖 AI 분석 실행'** 버튼을 눌러주세요.")
            st.markdown("""
#### AI 분석 제공 내용:
- ✅ 현재 차트 상황 요약
- ✅ 주목할 포인트 (2~3가지)
- ✅ 밸류에이션 (PER/PBR/배당) 코멘트
- ✅ 위험 요소
- ✅ 매수/매도 타이밍 제안
- ✅ 보수적 투자자 관점 조언
            """)


def _render_recommended_etfs():
    st.markdown("#### ⭐ 투자 성향별 추천 ETF")
    st.caption("📈 차트보기 버튼을 누르면 바로 해당 종목 차트로 이동해요.")

    style = st.radio(
        "투자 성향 선택",
        ["🛡️ 보수적", "⚖️ 중립 (적당히 공격적)"],
        horizontal=True,
        key="etf_style_radio",
    )

    if style == "🛡️ 보수적":
        RECOMMENDED = [
            ("🇰🇷 국내 ETF - 미국지수", "TIGER 미국S&P500", "360750.KS", "S&P500 추종. 원화로 미국 대형주 500개 분산투자. 장기 적립식 핵심"),
            ("🇰🇷 국내 ETF - 미국지수", "KODEX 미국S&P500TR", "379800.KS", "S&P500 배당 재투자형. 배당세 절약 효과"),
            ("🇰🇷 국내 ETF - 배당", "TIGER 미국배당다우존스", "458730.KS", "미국 배당성장주. 매월 배당 지급. 안정적 현금흐름"),
            ("🇰🇷 국내 ETF - 배당", "ACE 미국배당다우존스", "448540.KS", "TIGER와 유사. 매월 배당. 운용보수 비교 후 선택"),
            ("🇰🇷 국내 ETF - 안전자산", "TIGER KRX금현물", "411060.KS", "금 현물 추종. 인플레 헤지용. 포트폴리오 5~10% 권장"),
            ("🇰🇷 국내 ETF - 안전자산", "TIGER 미국채10년선물", "305080.KS", "미국 10년 국채. 주식 하락 시 방어 역할"),
            ("🇺🇸 미국 ETF", "VOO (S&P500)", "VOO", "뱅가드 S&P500. 운용보수 최저 수준. 미국 직접투자 핵심"),
            ("🇺🇸 미국 ETF", "SCHD (배당성장)", "SCHD", "미국 고품질 배당성장주. 장기 보유 시 배당+주가 동반 성장"),
            ("🇺🇸 미국 ETF", "JEPI (커버드콜)", "JEPI", "매월 높은 배당. 변동성 낮고 안정적 수입 원하는 분께"),
            ("🇺🇸 미국 ETF", "GLD (금)", "GLD", "금 ETF. 달러 약세·인플레 헤지용"),
        ]
        portfolio_tip = "💡 **보수적 포트폴리오 예시**\n\nS&P500 ETF 40% + 배당 ETF 30% + 금 10% + 채권 20%"

    else:
        RECOMMENDED = [
            ("🇰🇷 국내 ETF - 미국지수", "TIGER 미국나스닥100", "364980.KS", "나스닥100 추종. 기술주 비중 높음. S&P500보다 성장성 강함"),
            ("🇰🇷 국내 ETF - 미국지수", "KODEX 미국나스닥100TR", "379810.KS", "나스닥100 배당 재투자형. 배당세 절약"),
            ("🇰🇷 국내 ETF - 방산/테마", "PLUS K방산", "471540.KS", "한국 방산주 묶음. 글로벌 방산 수요 증가 수혜"),
            ("🇰🇷 국내 ETF - 방산/테마", "TIGER 방산", "425480.KS", "방산 테마. K-방산 수출 성장 수혜"),
            ("🇰🇷 국내 ETF - 반도체/AI", "TIGER 미국필라델피아반도체나스닥", "381180.KS", "미국 반도체 기업 집중. AI 수혜 섹터. 변동성 높음"),
            ("🇰🇷 국내 ETF - 반도체/AI", "KODEX AI반도체핵심장비", "461730.KS", "AI 반도체 장비주 집중. 고성장 기대 but 리스크 큼"),
            ("🇰🇷 국내 ETF - 글로벌", "TIGER 글로벌AI", "448920.KS", "글로벌 AI 기업 분산투자. 테마 ETF"),
            ("🇺🇸 미국 ETF", "QQQ (나스닥100)", "QQQ", "나스닥100 대표 ETF. 기술주 집중. 장기 성장성 높음"),
            ("🇺🇸 미국 ETF", "SOXX (반도체)", "SOXX", "미국 반도체 기업 집중. AI 수혜 최대 섹터"),
            ("🇺🇸 미국 ETF", "JEPQ (나스닥 커버드콜)", "JEPQ", "나스닥 기반 커버드콜. 높은 배당 + 기술주 노출"),
        ]
        portfolio_tip = "💡 **중립 포트폴리오 예시**\n\n나스닥 ETF 35% + 방산/AI 테마 25% + S&P500 25% + 금 15%"

    categories = {}
    for cat, name, ticker_r, desc in RECOMMENDED:
        categories.setdefault(cat, []).append((name, ticker_r, desc))

    for cat, items in categories.items():
        st.markdown(f"**{cat}**")
        for name, ticker_r, desc in items:
            c1, c2 = st.columns([3, 1])
            with c1:
                st.markdown(f"- **{name}** `{ticker_r}`  \n  {desc}")
            with c2:
                if st.button("📈 차트보기", key=f"rec_{ticker_r}"):
                    st.session_state["chart_ticker"] = ticker_r
                    st.session_state["nav_to"] = "📈 차트 분석"
                    st.rerun()
        st.divider()

    st.info(portfolio_tip)

    # 추천 보유 비율 파이차트
    st.markdown("---")
    st.markdown("#### 🥧 추천 보유 비율")
    if style == "🛡️ 보수적":
        labels = ["S&P500 ETF", "배당 ETF", "채권", "금"]
        values = [40, 30, 20, 10]
        colors = ["#29b6f6", "#4caf50", "#ff9800", "#ffd700"]
        desc_text = "안정성 최우선. 주식 70% + 안전자산 30% 구성."
    else:
        labels = ["나스닥 ETF", "방산/AI 테마", "S&P500 ETF", "금"]
        values = [35, 25, 25, 15]
        colors = ["#29b6f6", "#ef5350", "#4caf50", "#ffd700"]
        desc_text = "성장성 강화. 기술/테마 60% + 분산 40% 구성."

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.45,
        marker=dict(colors=colors),
        textinfo="label+percent",
        hovertemplate="%{label}: %{value}%<extra></extra>",
    ))
    fig.update_layout(
        height=320,
        paper_bgcolor="#131722",
        font=dict(color="#d1d4dc", size=13),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        margin=dict(l=10, r=10, t=20, b=40),
        annotations=[dict(text=f"{sum(values)}%", x=0.5, y=0.5, font_size=18, showarrow=False, font_color="#d1d4dc")],
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"💡 {desc_text}")
    st.warning("⚠️ 추천 ETF와 비율은 참고용입니다. 본인의 투자 목표와 상황에 맞게 판단하세요.")


def _render_trade_guide(signals: dict):
    score = signals.get("score", 0)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### 매수 고려 시점")
        st.markdown(f"""
- RSI **35 이하** 후 반등 전환
- 볼린저밴드 **하단**에서 반등
- MACD가 시그널선 **위로 돌파**
- MA20이 MA60 **위로 돌파** (골든크로스)
- 거래량 평균 **1.5배 이상** + 주가 상승

현재 점수: **{score:+d}점** {"→ 매수 우세" if score > 2 else ("→ 매도 우세" if score < -2 else "→ 중립")}
        """)
    with col_b:
        st.markdown("#### 매도 고려 시점")
        st.markdown("""
- RSI **70 이상** 후 하락 전환
- 볼린저밴드 **상단**에서 밀림
- MACD가 시그널선 **아래로 이탈**
- MA20이 MA60 **아래로 이탈** (데드크로스)
- 목표 수익률 달성 시 (+10~15%)
        """)
    st.warning("⚠️ 어떤 지표도 100% 정확하지 않습니다. 분할 매수/매도로 리스크 분산!")


def _render_news(news: list):
    if not news:
        st.info("최근 뉴스를 찾을 수 없습니다.")
        return
    for item in news[:8]:
        title = item.get("title", "제목 없음")
        link = item.get("link", "#")
        publisher = item.get("publisher", "")
        pub_time = item.get("providerPublishTime", 0)
        pub_date = (
            datetime.fromtimestamp(pub_time).strftime("%Y-%m-%d %H:%M")
            if pub_time
            else ""
        )
        st.markdown(f"**[{title}]({link})**")
        st.caption(f"{publisher}  |  {pub_date}")
        st.divider()


def _render_fundamentals(fundamentals: dict, ticker: str):
    st.markdown("#### 📊 기본 재무 지표")
    c1, c2 = st.columns(2)
    items = list(fundamentals.items())
    half = len(items) // 2 + len(items) % 2
    with c1:
        for k, v in items[:half]:
            st.markdown(f"**{k}**: {v}")
    with c2:
        for k, v in items[half:]:
            st.markdown(f"**{k}**: {v}")

    st.divider()
    st.markdown("#### 💡 지표 해석 가이드 (보수적 투자자용)")
    st.markdown("""
- **PER (주가수익비율)**: 낮을수록 저평가. 시장 평균 15~20 정도. **30 이상이면 고평가 주의**
- **PBR (주가순자산비율)**: 1 이하면 자산 대비 저평가. 성장주는 보통 높음
- **ROE (자기자본이익률)**: 10% 이상이면 양호, **15% 이상 우량**
- **배당수익률**: 2% 이상이면 인플레 대비 괜찮음, 4~5% 이상은 고배당주
- **배당성향**: 30~60% 권장. **100% 이상은 지속 가능성 위험**
- **베타**: 1이면 시장과 동일 변동, 1 이상이면 더 변동적 (**보수적이면 0.7~1.2** 선호)
    """)

    # 배당 히스토리
    dividends = get_dividends(ticker)
    if len(dividends) > 0:
        st.divider()
        st.markdown("#### 📅 배당 히스토리 (최근 5년)")
        recent_divs = dividends.tail(20)
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=recent_divs.index,
                y=recent_divs.values,
                marker_color="#4caf50",
                name="배당금",
            )
        )
        fig.update_layout(
            height=300,
            plot_bgcolor="#131722",
            paper_bgcolor="#131722",
            font=dict(color="#d1d4dc"),
            yaxis_title=f"배당금 ({currency_name(ticker)})",
            margin=dict(l=10, r=10, t=20, b=10),
        )
        fig.update_xaxes(gridcolor="#1e2130")
        fig.update_yaxes(gridcolor="#1e2130")
        st.plotly_chart(fig, use_container_width=True)


def _render_support_resistance(support: list, resistance: list, current: float, ticker: str):
    st.markdown("#### 🎯 자동 감지된 주요 가격대")
    st.caption("과거에 여러 번 막히거나 지지된 가격을 자동으로 찾아냅니다. 터치 횟수가 많을수록 강한 레벨입니다.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### 🔴 저항선 (매도 압력)")
        if not resistance:
            st.caption("감지된 저항선 없음")
        for price, touches in resistance:
            distance = (price - current) / current * 100
            st.markdown(
                f"- **{format_price(price, ticker)}** "
                f"({touches}회 터치) — 현재가 대비 **{distance:+.1f}%**"
            )
    with c2:
        st.markdown("##### 🟢 지지선 (매수 지원)")
        if not support:
            st.caption("감지된 지지선 없음")
        for price, touches in support:
            distance = (price - current) / current * 100
            st.markdown(
                f"- **{format_price(price, ticker)}** "
                f"({touches}회 터치) — 현재가 대비 **{distance:+.1f}%**"
            )

    st.info("""
💡 **활용법**
- 현재가가 **지지선 근처**에서 반등하면 매수 시점 고려
- 현재가가 **저항선 근처**에서 꺾이면 매도 시점 고려
- 저항선을 **뚫고 올라가면** 그 선이 새로운 지지선이 됨 (역할 반전)
    """)


def _landing_help():
    st.info("👈 왼쪽 사이드바에서 종목을 선택하거나 입력하세요.")
    col1, col2 = st.columns(2)
    with col1:
        with st.expander("📚 기술적 지표 설명", expanded=True):
            st.markdown("""
**이동평균선 (MA)**
- MA20 주황 / MA60 초록 / MA120 보라
- 골든크로스(MA20↗MA60) → 🟢 매수
- 데드크로스(MA20↘MA60) → 🔴 매도

**RSI (상대강도지수)**
- 70+ → 과매수
- 30- → 과매도

**볼린저밴드**
- 상단 = 저항, 하단 = 지지

**MACD**
- 시그널선 돌파 = 매수/매도 신호
            """)
    with col2:
        with st.expander("💡 보수적 투자 원칙", expanded=True):
            st.markdown("""
1. **최저/최고점 노리지 않기** — 추세에 올라타자
2. **분할 매수/매도** — 3번에 나눠 매수
3. **매수 조건 2~3개 동시 충족 시 진입**
4. **매도 조건 2~3개 동시 충족 시 정리**
5. **손절: 매수가 대비 -7~10% 하락 시 검토**
            """)


# =========================
# 2. 관심종목 대시보드
# =========================
def page_watchlist():
    st.title("⭐ 관심종목 대시보드")
    st.caption("저장한 종목의 현재 신호를 한눈에 비교")

    watchlist = load_watchlist()
    if not watchlist:
        st.info("저장된 관심종목이 없습니다. **차트 분석 페이지에서 종목 추가하기** 버튼을 눌러주세요.")
        return

    with st.spinner(f"{len(watchlist)}개 종목 분석 중..."):
        rows = []
        for w in watchlist:
            ticker = w["ticker"]
            df, info = get_stock_data(ticker, "3mo")
            if df.empty:
                continue
            df = calculate_indicators(df)
            sigs = analyze_signals(df)
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else latest
            change_pct = (latest["Close"] - prev["Close"]) / prev["Close"] * 100

            prev_week = df.iloc[-5] if len(df) > 5 else df.iloc[0]
            week_pct = (latest["Close"] - prev_week["Close"]) / prev_week["Close"] * 100

            fund = extract_fundamentals(info)

            rows.append(
                {
                    "종목": w.get("name", ticker)[:25],
                    "티커": ticker,
                    "현재가": format_price(latest["Close"], ticker),
                    "1일%": round(change_pct, 2),
                    "1주%": round(week_pct, 2),
                    "RSI": sigs.get("rsi", {}).get("value", "-"),
                    "종합신호": sigs.get("overall", "-"),
                    "점수": sigs.get("score", 0),
                    "PER": fund.get("PER (trailing)", "-"),
                    "배당%": fund.get("배당수익률", "-"),
                }
            )

    if not rows:
        st.warning("데이터를 가져올 수 있는 종목이 없습니다.")
        return

    df_wl = pd.DataFrame(rows)
    df_wl = df_wl.sort_values("점수", ascending=False).reset_index(drop=True)

    # 스타일 적용
    def score_color(val):
        try:
            v = int(val)
            if v >= 3:
                return "background-color: #1b5e20; color: white"
            if v >= 1:
                return "background-color: #2e7d32; color: white"
            if v <= -3:
                return "background-color: #b71c1c; color: white"
            if v <= -1:
                return "background-color: #c62828; color: white"
        except Exception:
            pass
        return ""

    def pct_color(val):
        try:
            v = float(val)
            if v > 0:
                return "color: #ef5350"
            if v < 0:
                return "color: #1565c0"
        except Exception:
            pass
        return ""

    styled = df_wl.style.map(score_color, subset=["점수"]).map(
        pct_color, subset=["1일%", "1주%"]
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### 🗑️ 관심종목 관리")
    col1, col2 = st.columns([2, 1])
    with col1:
        to_remove = st.selectbox(
            "제거할 종목",
            [""] + [f"{w.get('name', w['ticker'])} ({w['ticker']})" for w in watchlist],
        )
    with col2:
        st.write("")
        st.write("")
        if st.button("제거", type="secondary", use_container_width=True) and to_remove:
            ticker = to_remove.split("(")[-1].rstrip(")")
            remove_from_watchlist(ticker)
            st.success(f"{ticker} 제거됨")
            st.rerun()


# =========================
# 3. 매수 시뮬레이터
# =========================
def page_simulator():
    st.title("🧮 매수 시뮬레이터")
    st.caption("'그때 샀으면 얼마가 됐을까?' 계산기 - 타이밍 감각 익히기")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("📌 시뮬레이션 조건")
        ticker = stock_picker("sim")
        if not ticker:
            st.info("종목을 선택하세요.")
            return

        today = date.today()
        purchase_date = st.date_input(
            "매수 날짜",
            value=today - timedelta(days=365),
            max_value=today - timedelta(days=1),
            min_value=today - timedelta(days=365 * 20),
        )
        amount = st.number_input(
            f"투자 금액 ({currency_name(ticker)})",
            min_value=0.0,
            value=1000000.0 if is_korean(ticker) else 1000.0,
            step=100000.0 if is_korean(ticker) else 100.0,
        )

    purchase_date_str = purchase_date.strftime("%Y-%m-%d")
    buy_price = get_price_on_date(ticker, purchase_date_str)
    current_df, info = get_stock_data(ticker, "5y")
    if current_df.empty or buy_price is None:
        st.error("해당 날짜의 가격 데이터를 가져올 수 없습니다.")
        return

    current_price = float(current_df["Close"].iloc[-1])
    shares = amount / buy_price
    current_value = shares * current_price
    profit = current_value - amount
    profit_pct = profit / amount * 100

    days = (today - purchase_date).days
    years = days / 365.25
    annualized = ((current_value / amount) ** (1 / years) - 1) * 100 if years > 0 else 0

    with col2:
        st.subheader("📊 결과")
        st.metric("매수가", format_price(buy_price, ticker))
        st.metric("현재가", format_price(current_price, ticker))
        st.metric("매수 수량", f"{shares:.4f} 주")
        st.metric(
            "현재 평가금액",
            format_price(current_value, ticker),
            f"{profit_pct:+.2f}%",
        )
        st.metric(
            "총 수익금",
            format_price(profit, ticker),
            f"{years:.1f}년 보유",
        )
        st.metric("연환산 수익률", f"{annualized:+.2f}%")

    # 수익률 추이 차트
    st.divider()
    st.subheader("📈 보유 기간 동안의 평가금액 추이")

    mask = current_df.index >= pd.Timestamp(purchase_date)
    period_df = current_df[mask].copy()
    period_df["평가금액"] = shares * period_df["Close"]
    period_df["수익률"] = (period_df["Close"] / buy_price - 1) * 100

    fig = make_dual_axis_chart(period_df, ticker, amount)
    st.plotly_chart(fig, use_container_width=True)

    # 최고/최저 시점
    max_idx = period_df["평가금액"].idxmax()
    min_idx = period_df["평가금액"].idxmin()
    c1, c2 = st.columns(2)
    with c1:
        max_val = period_df.loc[max_idx, "평가금액"]
        max_pct = period_df.loc[max_idx, "수익률"]
        st.success(
            f"📈 **최고 평가금액**: {format_price(max_val, ticker)} "
            f"({max_pct:+.2f}%)\n\n{max_idx.strftime('%Y-%m-%d')}"
        )
    with c2:
        min_val = period_df.loc[min_idx, "평가금액"]
        min_pct = period_df.loc[min_idx, "수익률"]
        st.error(
            f"📉 **최저 평가금액**: {format_price(min_val, ticker)} "
            f"({min_pct:+.2f}%)\n\n{min_idx.strftime('%Y-%m-%d')}"
        )

    st.info("💡 이 시뮬레이션은 **가상의 매수**입니다. 실제 매수 시 수수료/세금은 포함되어 있지 않습니다.")


def make_dual_axis_chart(period_df: pd.DataFrame, ticker: str, initial: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=period_df.index,
            y=period_df["평가금액"],
            name="평가금액",
            line=dict(color="#29b6f6", width=2),
            fill="tozeroy",
            fillcolor="rgba(41,182,246,0.1)",
        )
    )
    fig.add_hline(
        y=initial,
        line_dash="dash",
        line_color="#ff9800",
        annotation_text=f"원금 {format_price(initial, ticker)}",
        annotation_position="right",
    )
    fig.update_layout(
        height=400,
        plot_bgcolor="#131722",
        paper_bgcolor="#131722",
        font=dict(color="#d1d4dc"),
        yaxis_title=f"평가금액 ({currency_name(ticker)})",
        margin=dict(l=10, r=10, t=20, b=10),
    )
    fig.update_xaxes(gridcolor="#1e2130")
    fig.update_yaxes(gridcolor="#1e2130")
    return fig


# =========================
# 4. 매매 일지 + AI 코칭
# =========================
def page_trade_journal():
    st.title("📓 매매 일지")
    st.caption("내 매매 기록을 남기고, AI 코치에게 내 매매 패턴을 분석받기")

    tab_add, tab_list, tab_analysis = st.tabs(["➕ 매매 추가", "📋 기록 목록", "🤖 AI 코칭"])

    with tab_add:
        _trade_add_form()
    with tab_list:
        _trade_list()
    with tab_analysis:
        _trade_ai_analysis()


def _trade_add_form():
    st.subheader("새 매매 기록")

    ticker = stock_picker("trade")
    if not ticker:
        st.info("종목을 먼저 선택하세요.")
        return

    c1, c2 = st.columns(2)
    with c1:
        action = st.radio("매매 종류", ["매수", "매도"], horizontal=True)
        trade_date = st.date_input(
            "매매 날짜",
            value=date.today(),
            max_value=date.today(),
        )
    with c2:
        # 자동으로 해당 날짜 가격 가져와서 기본값 넣어줌
        price_on_date = get_price_on_date(ticker, trade_date.strftime("%Y-%m-%d"))
        default_price = price_on_date if price_on_date else 0.0
        price = st.number_input(
            f"매매 가격 ({currency_name(ticker)})",
            min_value=0.0,
            value=float(default_price),
            step=1.0 if is_korean(ticker) else 0.01,
        )
        if price_on_date:
            st.caption(f"💡 해당 날짜 종가: {format_price(price_on_date, ticker)} (참고)")
        qty = st.number_input("수량 (주)", min_value=0.0, value=1.0, step=0.1)

    reason = st.text_area(
        "매매 이유 (왜 이 시점에 샀는지/팔았는지)",
        placeholder="예: RSI 30 아래로 떨어지고 볼린저밴드 하단 터치해서 반등 기대 / 실적 발표 앞두고 불안해서 일부 정리",
        height=100,
    )

    # 당시 기술적 지표 스냅샷 계산 (선택적 자동 저장)
    with st.expander("📸 당시 차트 상황 (자동 기록)"):
        with st.spinner("해당 시점 기술적 지표 분석 중..."):
            snapshot = _get_snapshot(ticker, trade_date)
        if snapshot:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**RSI**: {snapshot.get('rsi', 'N/A')}")
                st.markdown(f"**MACD**: {snapshot.get('macd_signal', 'N/A')}")
            with c2:
                st.markdown(f"**MA 추세**: {snapshot.get('ma_trend', 'N/A')}")
                st.markdown(f"**종합점수**: {snapshot.get('score', 'N/A')}")
        else:
            st.caption("해당 날짜의 데이터를 가져올 수 없습니다.")

    if st.button("💾 매매 기록 저장", type="primary", use_container_width=True):
        if price <= 0 or qty <= 0:
            st.error("가격과 수량은 0보다 커야 합니다.")
            return
        if not reason.strip():
            st.warning("매매 이유를 적어야 나중에 AI 분석이 의미 있어요. 그래도 저장할까요?")
        trade = {
            "ticker": ticker,
            "name": TICKER_TO_NAME.get(ticker, ticker),
            "action": action,
            "date": trade_date.strftime("%Y-%m-%d"),
            "price": float(price),
            "qty": float(qty),
            "reason": reason.strip() or "(이유 미기재)",
            "context": snapshot or {},
        }
        add_trade(trade)
        st.success(f"✅ 저장 완료! {trade_date} {ticker} {action} {qty}주 @ {price:,.2f}")
        st.balloons()


def _get_snapshot(ticker: str, trade_date: date) -> dict | None:
    """해당 날짜 시점의 기술적 지표 스냅샷"""
    try:
        # 해당 날짜까지 6개월 데이터 확보 (지표 계산 위해)
        end = trade_date + timedelta(days=1)
        start = trade_date - timedelta(days=200)
        import yfinance as yf
        df = yf.Ticker(ticker).history(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))
        if df.empty:
            return None
        if df.index.tzinfo is not None:
            df.index = df.index.tz_localize(None)
        df = calculate_indicators(df)
        sigs = analyze_signals(df)
        rsi_val = sigs.get("rsi", {}).get("value")
        ma20 = df["MA20"].iloc[-1] if "MA20" in df.columns else None
        ma60 = df["MA60"].iloc[-1] if "MA60" in df.columns else None
        ma_trend = "상승" if (ma20 and ma60 and ma20 > ma60) else "하락"
        return {
            "rsi": rsi_val,
            "macd_signal": sigs.get("macd", {}).get("signal", "N/A"),
            "bb_signal": sigs.get("bollinger", {}).get("signal", "N/A"),
            "ma_trend": ma_trend,
            "score": sigs.get("score", 0),
            "overall": sigs.get("overall", "N/A"),
        }
    except Exception:
        return None


def _trade_list():
    trades = load_trades()
    if not trades:
        st.info("저장된 매매 기록이 없습니다. '매매 추가' 탭에서 기록을 남겨보세요.")
        return

    st.subheader(f"📋 총 {len(trades)}건의 매매 기록")

    # 필터
    c1, c2 = st.columns(2)
    with c1:
        all_tickers = sorted({t["ticker"] for t in trades})
        filter_ticker = st.selectbox("종목 필터", ["전체"] + all_tickers)
    with c2:
        filter_action = st.selectbox("행동 필터", ["전체", "매수", "매도"])

    filtered = trades
    if filter_ticker != "전체":
        filtered = [t for t in filtered if t["ticker"] == filter_ticker]
    if filter_action != "전체":
        filtered = [t for t in filtered if t["action"] == filter_action]

    # 최신순 정렬
    filtered = sorted(filtered, key=lambda t: t.get("date", ""), reverse=True)

    for trade in filtered:
        action_icon = "🟢 매수" if trade["action"] == "매수" else "🔴 매도"
        price_str = format_price(trade["price"], trade["ticker"])
        total = trade["price"] * trade["qty"]
        total_str = format_price(total, trade["ticker"])

        with st.expander(
            f"{trade['date']}  |  {action_icon}  |  **{trade['ticker']}**  "
            f"{trade['qty']}주 @ {price_str}  (총 {total_str})"
        ):
            st.markdown(f"**📝 매매 이유**")
            st.markdown(f"> {trade.get('reason', '(없음)')}")

            ctx = trade.get("context", {})
            if ctx:
                st.markdown("**📸 당시 차트 상황**")
                c1, c2 = st.columns(2)
                with c1:
                    st.caption(f"RSI: **{ctx.get('rsi', 'N/A')}**")
                    st.caption(f"MACD: {ctx.get('macd_signal', 'N/A')}")
                    st.caption(f"볼린저: {ctx.get('bb_signal', 'N/A')}")
                with c2:
                    st.caption(f"MA 추세: **{ctx.get('ma_trend', 'N/A')}**")
                    st.caption(f"종합 점수: **{ctx.get('score', 0):+}**")
                    st.caption(f"종합 신호: {ctx.get('overall', 'N/A')}")

            b1, b2, _ = st.columns([1, 1, 3])
            trade_key = trade.get("id", f"{trade.get('date', '')}_{trade.get('ticker', '')}")
            with b1:
                if st.button("🤖 이 매매 AI 분석", key=f"ai_{trade_key}"):
                    with st.spinner("분석 중..."):
                        import yfinance as yf
                        start = (pd.to_datetime(trade["date"]) - timedelta(days=200)).strftime("%Y-%m-%d")
                        end = (pd.to_datetime(trade["date"]) + timedelta(days=1)).strftime("%Y-%m-%d")
                        context_df = yf.Ticker(trade["ticker"]).history(start=start, end=end)
                        if context_df.index.tzinfo is not None:
                            context_df.index = context_df.index.tz_localize(None)
                        result = ai_analyze_single_trade(trade, context_df)
                    st.markdown(result)
            with b2:
                if st.button("🗑️ 삭제", key=f"del_{trade_key}"):
                    delete_trade(trade.get("id", ""))
                    st.success("삭제 완료")
                    st.rerun()


def _trade_ai_analysis():
    trades = load_trades()
    if not trades:
        st.info("매매 기록이 있어야 분석이 가능합니다.")
        return

    st.markdown(f"#### 🤖 AI 코치에게 내 매매 패턴 분석받기")
    st.caption(f"지금까지 {len(trades)}건의 매매 기록을 바탕으로 분석합니다.")

    st.markdown("""
**AI가 분석해주는 것:**
- ✅ 내 매매 스타일의 전반적 평가
- ✅ 잘하고 있는 부분 (구체적 거래 예시)
- ✅ 고칠 점 / 주의할 패턴
- ✅ 매수/매도 이유가 차트 상황과 맞았는지
- ✅ 다음 매매 전 체크 리스트
    """)

    if st.button("🚀 전체 매매 패턴 분석 실행", type="primary", use_container_width=True):
        with st.spinner("Claude AI가 분석 중입니다... (20~40초)"):
            result = ai_analyze_trades(trades)
        st.markdown(result)
        st.divider()
        st.caption("💡 분석 결과를 캡처해 두고 다음 매매에 참고하세요!")


# =========================
# 5. 내 포트폴리오 (한투 연동)
# =========================
def page_portfolio():
    from kis_api import get_access_token, get_portfolio, is_configured

    st.title("💼 내 포트폴리오")
    st.caption("한국투자증권 계좌 실시간 연동")

    if not is_configured():
        st.error("한국투자증권 API 키가 설정되지 않았습니다.")
        st.markdown("""
**.env 파일에 아래 3가지를 입력하세요:**
```
KIS_APP_KEY=발급받은앱키
KIS_APP_SECRET=발급받은시크릿
KIS_ACCOUNT_NO=계좌번호8자리
```
        """)
        return

    col_refresh, _ = st.columns([1, 4])
    with col_refresh:
        if st.button("🔄 새로고침", use_container_width=True):
            get_portfolio.clear()
            get_access_token.clear()
            st.rerun()

    with st.spinner("계좌 정보 불러오는 중..."):
        holdings, summary = get_portfolio()

    if "error" in summary:
        st.error(f"오류: {summary['error']}")
        st.info("API 키나 계좌번호를 다시 확인해보세요.")
        return

    if not holdings and not summary:
        st.warning("데이터를 불러오지 못했습니다. 잠시 후 새로고침 해보세요.")
        return

    # ── 계좌 요약 ──
    if summary:
        st.subheader("📊 계좌 요약")
        m1, m2, m3, m4 = st.columns(4)
        total_eval = summary.get("total_eval", 0)
        total_purchase = summary.get("total_purchase", 0)
        total_pl = summary.get("total_profit_loss", 0)
        pl_pct = summary.get("profit_loss_pct", 0)
        cash = summary.get("cash", 0)

        with m1:
            st.metric("총 평가금액", f"₩{total_eval:,.0f}")
        with m2:
            st.metric("총 매입금액", f"₩{total_purchase:,.0f}")
        with m3:
            st.metric("총 수익/손실", f"₩{total_pl:,.0f}", f"{pl_pct:+.2f}%")
        with m4:
            st.metric("예수금 (현금)", f"₩{cash:,.0f}")

        st.divider()

    # ── 비율 파이차트 ──
    st.subheader("🥧 보유 비율")
    c_pie, c_bar = st.columns(2)
    with c_pie:
        fig_pie = go.Figure(go.Pie(
            labels=[h["name"] for h in holdings],
            values=[h["eval_amount"] for h in holdings],
            hole=0.4,
            textinfo="label+percent",
            hovertemplate="%{label}<br>₩%{value:,.0f}<br>%{percent}<extra></extra>",
        ))
        fig_pie.update_layout(
            height=350,
            paper_bgcolor="#131722",
            font=dict(color="#d1d4dc"),
            showlegend=False,
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    with c_bar:
        holdings_sorted_bar = sorted(holdings, key=lambda x: x["eval_amount"], reverse=True)
        fig_bar = go.Figure(go.Bar(
            x=[h["eval_amount"] for h in holdings_sorted_bar],
            y=[h["name"] for h in holdings_sorted_bar],
            orientation="h",
            marker_color=[
                "#ef5350" if h["profit_loss"] >= 0 else "#1565c0"
                for h in holdings_sorted_bar
            ],
            text=[f"₩{h['eval_amount']:,.0f}" for h in holdings_sorted_bar],
            textposition="auto",
        ))
        fig_bar.update_layout(
            height=350,
            paper_bgcolor="#131722",
            plot_bgcolor="#131722",
            font=dict(color="#d1d4dc"),
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis=dict(gridcolor="#1e2130"),
            yaxis=dict(gridcolor="#1e2130"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()

    # ── 보유 종목 ──
    st.subheader(f"📋 보유 종목 ({len(holdings)}개)")

    if not holdings:
        st.info("보유 종목이 없습니다.")
        return

    # 수익률 순 정렬
    holdings_sorted = sorted(holdings, key=lambda x: -x["profit_loss_pct"])

    for h in holdings_sorted:
        pl = h["profit_loss"]
        pl_pct = h["profit_loss_pct"]
        color = "🔴" if pl >= 0 else "🔵"
        sign = "▲" if pl >= 0 else "▼"

        with st.expander(
            f"{color} **{h['name']}**  |  "
            f"평가 ₩{h['eval_amount']:,.0f}  |  "
            f"{sign} {abs(pl_pct):.2f}%"
        ):
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("보유수량", f"{h['qty']:,.0f}주")
            with c2:
                st.metric("평균매수가", f"₩{h['avg_price']:,.0f}")
            with c3:
                st.metric("현재가", f"₩{h['current_price']:,.0f}")
            with c4:
                st.metric("수익/손실", f"₩{pl:,.0f}", f"{pl_pct:+.2f}%")

            b1, b2 = st.columns([1, 3])
            with b1:
                if st.button("📈 차트 분석", key=f"chart_{h['code']}"):
                    st.session_state["chart_ticker"] = h["ticker"]
                    st.session_state["nav_to"] = "📈 차트 분석"
                    st.rerun()
            with b2:
                st.caption(f"종목코드: {h['code']} | 티커: {h['ticker']}")


# =========================
# 6. 시장 지표
# =========================
def page_market():
    st.title("🌡️ 시장 지표")
    st.caption("시장 전체 분위기를 보고 투자 판단에 참고")

    c1, c2 = st.columns(2)
    with c1:
        _render_fear_greed()
    with c2:
        _render_exchange_rate()


def _render_fear_greed():
    st.subheader("😱 공포/탐욕 지수 (CNN)")
    with st.spinner("CNN Fear & Greed Index 가져오는 중..."):
        fg = get_fear_greed()

    if fg is None:
        st.warning("지수를 가져올 수 없습니다. 잠시 후 다시 시도해주세요.")
        return

    score = fg["score"]
    rating = fear_greed_korean(fg["rating"])

    if score < 25:
        color = "#1565c0"
    elif score < 45:
        color = "#29b6f6"
    elif score < 55:
        color = "#9e9e9e"
    elif score < 75:
        color = "#ef5350"
    else:
        color = "#c62828"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=score,
            title={"text": f"<b>{rating}</b>", "font": {"size": 18, "color": "white"}},
            delta={"reference": fg["prev_close"], "font": {"color": "white"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#d1d4dc"},
                "bar": {"color": color},
                "bgcolor": "#131722",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 25], "color": "rgba(21,101,192,0.3)"},
                    {"range": [25, 45], "color": "rgba(41,182,246,0.3)"},
                    {"range": [45, 55], "color": "rgba(158,158,158,0.3)"},
                    {"range": [55, 75], "color": "rgba(239,83,80,0.3)"},
                    {"range": [75, 100], "color": "rgba(198,40,40,0.3)"},
                ],
            },
            number={"font": {"color": "white", "size": 40}},
        )
    )
    fig.update_layout(
        height=320,
        paper_bgcolor="#131722",
        font=dict(color="white"),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("1주 전", f"{fg['prev_1w']:.1f}")
    c2.metric("1개월 전", f"{fg['prev_1m']:.1f}")
    c3.metric("전일 대비", f"{score - fg['prev_close']:+.1f}")

    st.markdown("""
**📖 지수 해석**
- **0~25 극심한 공포**: 시장 패닉 → 역사적으로 **매수 기회**가 많았던 구간
- **25~45 공포**: 조심스러운 시장 → 분할매수 고려
- **45~55 중립**: 방향성 불분명 → 관망
- **55~75 탐욕**: 낙관 분위기 → 추격매수 주의
- **75~100 극심한 탐욕**: 과열 → **매도/차익실현** 고려

> ⚠️ 미국 시장 기준입니다. 한국 시장과 다를 수 있음.
    """)


def _render_exchange_rate():
    st.subheader("💱 USD/KRW 환율")
    rate = get_exchange_rate()
    if rate is None:
        st.warning("환율 데이터를 가져올 수 없습니다.")
        return

    # 과거 환율 트렌드
    import yfinance as yf
    df = yf.Ticker("KRW=X").history(period="6mo")
    if not df.empty and df.index.tzinfo is not None:
        df.index = df.index.tz_localize(None)

    fig = go.Figure()
    if not df.empty:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Close"],
                name="USD/KRW",
                line=dict(color="#29b6f6", width=2),
                fill="tozeroy",
                fillcolor="rgba(41,182,246,0.1)",
            )
        )
    fig.update_layout(
        height=320,
        plot_bgcolor="#131722",
        paper_bgcolor="#131722",
        font=dict(color="#d1d4dc"),
        yaxis_title="원/달러",
        margin=dict(l=10, r=10, t=40, b=10),
        title=dict(text=f"<b>현재: {rate:,.2f} 원</b>", font=dict(size=16)),
    )
    fig.update_xaxes(gridcolor="#1e2130")
    fig.update_yaxes(gridcolor="#1e2130")
    st.plotly_chart(fig, use_container_width=True)

    if not df.empty and len(df) > 1:
        pct_1m = (rate - df["Close"].iloc[-20]) / df["Close"].iloc[-20] * 100 if len(df) > 20 else 0
        pct_3m = (rate - df["Close"].iloc[0]) / df["Close"].iloc[0] * 100
        c1, c2 = st.columns(2)
        c1.metric("1개월 변동", f"{pct_1m:+.2f}%")
        c2.metric("6개월 변동", f"{pct_3m:+.2f}%")

    st.markdown("""
**💡 환율이 미국주식 수익에 미치는 영향**
- **원/달러 상승 (원화 약세)** → 환차익 → 수익 추가 발생
- **원/달러 하락 (원화 강세)** → 환차손 → 주가가 올라도 수익 줄어듦
- 미국주식 장기 보유할 땐 환율까지 고려한 수익률로 봐야 정확함
    """)


