"""
주식 차트 분석 - 코어 로직
- 데이터 가져오기 (yfinance)
- 기술적 지표 계산 (ta)
- 매수/매도 신호 분석
- 지지/저항선 감지
- 차트 생성 (plotly)
- 관심종목/매매일지 저장 (JSON)
- AI 분석 (Claude)
"""

from __future__ import annotations

import json
import os
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import anthropic
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import ta
import yfinance as yf
from dotenv import load_dotenv
from plotly.subplots import make_subplots

from constants import currency_name, currency_symbol, is_korean, format_price

warnings.filterwarnings("ignore")
load_dotenv()

# =========================
# 저장 경로 설정
# =========================
PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

WATCHLIST_FILE = DATA_DIR / "watchlist.json"
TRADES_FILE = DATA_DIR / "trades.json"


# =========================
# 데이터 가져오기
# =========================
@st.cache_data(ttl=300)
def get_stock_data(ticker: str, period: str = "1y") -> tuple[pd.DataFrame, dict]:
    """주가 데이터 + 종목 정보 조회"""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        if df.empty:
            return pd.DataFrame(), {}
        if df.index.tzinfo is not None:
            df.index = df.index.tz_localize(None)
        info = stock.info or {}
        return df, info
    except Exception:
        return pd.DataFrame(), {}


@st.cache_data(ttl=1800)
def get_news(ticker: str) -> list:
    """최근 뉴스 목록"""
    try:
        stock = yf.Ticker(ticker)
        return stock.news[:10] if stock.news else []
    except Exception:
        return []


@st.cache_data(ttl=3600)
def get_dividends(ticker: str) -> pd.Series:
    """배당 히스토리"""
    try:
        stock = yf.Ticker(ticker)
        div = stock.dividends
        if len(div) == 0:
            return pd.Series(dtype=float)
        if div.index.tzinfo is not None:
            div.index = div.index.tz_localize(None)
        return div
    except Exception:
        return pd.Series(dtype=float)


@st.cache_data(ttl=3600)
def get_exchange_rate() -> float | None:
    """USD/KRW 환율"""
    try:
        rate = yf.Ticker("KRW=X").history(period="5d")
        if rate.empty:
            return None
        return float(rate["Close"].iloc[-1])
    except Exception:
        return None


@st.cache_data(ttl=1800)
def get_fear_greed() -> dict | None:
    """CNN Fear & Greed 지수"""
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
        }
        url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        fg = data.get("fear_and_greed", {})
        return {
            "score": float(fg.get("score", 0)),
            "rating": fg.get("rating", "unknown"),
            "prev_close": float(fg.get("previous_close", 0)),
            "prev_1w": float(fg.get("previous_1_week", 0)),
            "prev_1m": float(fg.get("previous_1_month", 0)),
        }
    except Exception:
        return None


def fear_greed_korean(rating: str) -> str:
    mapping = {
        "extreme fear": "극심한 공포 🥶",
        "fear": "공포 😨",
        "neutral": "중립 😐",
        "greed": "탐욕 😊",
        "extreme greed": "극심한 탐욕 🤑",
    }
    return mapping.get(rating.lower(), rating)


# =========================
# 기술적 지표 계산
# =========================
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 20:
        return df
    try:
        df["MA20"] = ta.trend.sma_indicator(df["Close"], window=20)
        if len(df) >= 60:
            df["MA60"] = ta.trend.sma_indicator(df["Close"], window=60)
        if len(df) >= 120:
            df["MA120"] = ta.trend.sma_indicator(df["Close"], window=120)

        bb = ta.volatility.BollingerBands(df["Close"], window=20, window_dev=2)
        df["BB_upper"] = bb.bollinger_hband()
        df["BB_middle"] = bb.bollinger_mavg()
        df["BB_lower"] = bb.bollinger_lband()

        df["RSI"] = ta.momentum.rsi(df["Close"], window=14)

        macd = ta.trend.MACD(df["Close"])
        df["MACD"] = macd.macd()
        df["MACD_signal"] = macd.macd_signal()
        df["MACD_hist"] = macd.macd_diff()
    except Exception:
        pass
    return df


# =========================
# 지지선 / 저항선 감지
# =========================
def find_support_resistance(
    df: pd.DataFrame, window: int = 5, tolerance: float = 0.015, max_levels: int = 4
) -> tuple[list, list]:
    """
    피봇 고점/저점 찾고 가까운 가격대끼리 묶어서 지지/저항선 반환.
    각 요소: (가격, 터치횟수) 튜플.
    """
    if len(df) < window * 2 + 1:
        return [], []

    highs = df["High"].values
    lows = df["Low"].values
    pivot_highs = []
    pivot_lows = []

    for i in range(window, len(df) - window):
        if all(highs[i] >= highs[i - j] for j in range(1, window + 1)) and all(
            highs[i] >= highs[i + j] for j in range(1, window + 1)
        ):
            pivot_highs.append(highs[i])
        if all(lows[i] <= lows[i - j] for j in range(1, window + 1)) and all(
            lows[i] <= lows[i + j] for j in range(1, window + 1)
        ):
            pivot_lows.append(lows[i])

    def cluster(levels: list) -> list:
        if not levels:
            return []
        levels = sorted(levels)
        clusters = [[levels[0]]]
        for lv in levels[1:]:
            avg = sum(clusters[-1]) / len(clusters[-1])
            if abs(lv - avg) / avg < tolerance:
                clusters[-1].append(lv)
            else:
                clusters.append([lv])
        result = [(sum(c) / len(c), len(c)) for c in clusters if len(c) >= 1]
        # 터치횟수 많은 순으로 정렬, 상위 max_levels개
        result.sort(key=lambda x: -x[1])
        return result[:max_levels]

    resistance = cluster(pivot_highs)
    support = cluster(pivot_lows)
    return support, resistance


# =========================
# 매수/매도 신호 분석
# =========================
def analyze_signals(df: pd.DataFrame) -> dict:
    if df.empty or len(df) < 20:
        return {"overall": "데이터 부족", "score": 0}

    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest

    signals = {
        "rsi": {"value": None, "signal": "계산 불가"},
        "macd": {"signal": "계산 불가"},
        "bollinger": {"signal": "계산 불가"},
        "ma": {"signal": "계산 불가"},
        "overall": "중립",
        "score": 0,
    }
    score = 0

    # RSI
    if "RSI" in df.columns and pd.notna(latest.get("RSI")):
        rsi = latest["RSI"]
        signals["rsi"]["value"] = round(rsi, 1)
        if rsi < 30:
            signals["rsi"]["signal"] = "과매도 → 매수 고려"
            score += 2
        elif rsi < 40:
            signals["rsi"]["signal"] = f"약한 과매도 ({rsi:.0f})"
            score += 1
        elif rsi > 70:
            signals["rsi"]["signal"] = "과매수 → 매도 고려"
            score -= 2
        elif rsi > 60:
            signals["rsi"]["signal"] = f"약한 과매수 ({rsi:.0f})"
            score -= 1
        else:
            signals["rsi"]["signal"] = f"중립 ({rsi:.0f})"

    # MACD
    if all(c in df.columns for c in ["MACD", "MACD_signal"]):
        m, ms = latest.get("MACD"), latest.get("MACD_signal")
        pm, pms = prev.get("MACD"), prev.get("MACD_signal")
        if all(pd.notna(x) for x in [m, ms, pm, pms]):
            curr_diff = m - ms
            prev_diff = pm - pms
            if curr_diff > 0 and prev_diff <= 0:
                signals["macd"]["signal"] = "골든크로스 (강한 매수)"
                score += 3
            elif curr_diff < 0 and prev_diff >= 0:
                signals["macd"]["signal"] = "데드크로스 (강한 매도)"
                score -= 3
            elif curr_diff > 0:
                signals["macd"]["signal"] = "매수 구간 유지"
                score += 1
            else:
                signals["macd"]["signal"] = "매도 구간 유지"
                score -= 1

    # 볼린저밴드
    if all(c in df.columns for c in ["BB_upper", "BB_lower", "BB_middle"]):
        bu, bl, bm = latest.get("BB_upper"), latest.get("BB_lower"), latest.get("BB_middle")
        if all(pd.notna(x) for x in [bu, bl, bm]):
            close = latest["Close"]
            if close <= bl:
                signals["bollinger"]["signal"] = "하단 터치 → 반등 가능성"
                score += 2
            elif close >= bu:
                signals["bollinger"]["signal"] = "상단 터치 → 조정 가능성"
                score -= 2
            elif close > bm:
                signals["bollinger"]["signal"] = "중간선 위 (긍정적)"
                score += 1
            else:
                signals["bollinger"]["signal"] = "중간선 아래 (부정적)"
                score -= 1

    # 이동평균
    if "MA20" in df.columns and "MA60" in df.columns:
        ma20, ma60 = latest.get("MA20"), latest.get("MA60")
        pma20, pma60 = prev.get("MA20"), prev.get("MA60")
        if all(pd.notna(x) for x in [ma20, ma60, pma20, pma60]):
            curr_diff = ma20 - ma60
            prev_diff = pma20 - pma60
            if curr_diff > 0 and prev_diff <= 0:
                signals["ma"]["signal"] = "골든크로스 발생! (강한 매수)"
                score += 4
            elif curr_diff < 0 and prev_diff >= 0:
                signals["ma"]["signal"] = "데드크로스 발생! (강한 매도)"
                score -= 4
            elif curr_diff > 0:
                signals["ma"]["signal"] = "상승 추세 (MA20 > MA60)"
                score += 1
            else:
                signals["ma"]["signal"] = "하락 추세 (MA20 < MA60)"
                score -= 1

    signals["score"] = score
    if score >= 5:
        signals["overall"] = "강한 매수 신호"
    elif score >= 2:
        signals["overall"] = "약한 매수 신호"
    elif score <= -5:
        signals["overall"] = "강한 매도 신호"
    elif score <= -2:
        signals["overall"] = "약한 매도 신호"
    else:
        signals["overall"] = "관망 (중립)"
    return signals


# =========================
# 기본 재무 지표 추출
# =========================
def extract_fundamentals(info: dict) -> dict:
    """info 딕셔너리에서 쓸만한 재무지표 추출"""

    def pct(x):
        return f"{x*100:.2f}%" if x is not None else "N/A"

    def num(x, fmt=","):
        if x is None:
            return "N/A"
        try:
            return format(float(x), fmt)
        except Exception:
            return str(x)

    market_cap = info.get("marketCap")
    if market_cap:
        if market_cap > 1e12:
            mc_str = f"{market_cap/1e12:.2f}조 (T)"
        elif market_cap > 1e9:
            mc_str = f"{market_cap/1e9:.2f}B (10억)"
        else:
            mc_str = f"{market_cap/1e6:.2f}M (백만)"
    else:
        mc_str = "N/A"

    return {
        "PER (trailing)": num(info.get("trailingPE"), ".2f"),
        "PER (forward)": num(info.get("forwardPE"), ".2f"),
        "PBR": num(info.get("priceToBook"), ".2f"),
        "ROE": pct(info.get("returnOnEquity")),
        "배당수익률": pct(info.get("dividendYield")),
        "배당금(연)": num(info.get("dividendRate"), ".2f"),
        "배당성향": pct(info.get("payoutRatio")),
        "시가총액": mc_str,
        "52주 최고": num(info.get("fiftyTwoWeekHigh"), ",.2f"),
        "52주 최저": num(info.get("fiftyTwoWeekLow"), ",.2f"),
        "베타": num(info.get("beta"), ".2f"),
        "섹터": info.get("sector", "N/A"),
        "산업": info.get("industry", "N/A"),
    }


# =========================
# 차트 생성
# =========================
def create_chart(
    df: pd.DataFrame,
    show_bb: bool = True,
    show_ma: bool = True,
    show_sr: bool = True,
    support: list | None = None,
    resistance: list | None = None,
    trade_markers: list | None = None,
) -> go.Figure:
    """메인 캔들 차트"""
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.52, 0.16, 0.16, 0.16],
        subplot_titles=("캔들 차트", "거래량", "RSI (14)", "MACD"),
    )

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="가격",
            increasing=dict(line=dict(color="#ef5350"), fillcolor="#ef5350"),
            decreasing=dict(line=dict(color="#1565c0"), fillcolor="#1565c0"),
        ),
        row=1,
        col=1,
    )

    if show_ma:
        for col_name, color, label in [
            ("MA20", "#ff9800", "MA20(단기)"),
            ("MA60", "#4caf50", "MA60(중기)"),
            ("MA120", "#9c27b0", "MA120(장기)"),
        ]:
            if col_name in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df[col_name],
                        name=label,
                        line=dict(color=color, width=1.5),
                        opacity=0.9,
                    ),
                    row=1,
                    col=1,
                )

    if show_bb and "BB_upper" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["BB_upper"],
                name="BB상단",
                line=dict(color="#78909c", width=1, dash="dot"),
                opacity=0.6,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["BB_lower"],
                name="BB하단",
                line=dict(color="#78909c", width=1, dash="dot"),
                opacity=0.6,
                fill="tonexty",
                fillcolor="rgba(120,144,156,0.08)",
            ),
            row=1,
            col=1,
        )

    # 지지/저항선
    if show_sr:
        if resistance:
            for price, touches in resistance:
                fig.add_hline(
                    y=price,
                    line=dict(color="rgba(239,83,80,0.45)", width=1, dash="dash"),
                    annotation=dict(
                        text=f"저항 {price:,.0f} ({touches}회)",
                        font=dict(color="#ef5350", size=10),
                    ),
                    annotation_position="right",
                    row=1,
                    col=1,
                )
        if support:
            for price, touches in support:
                fig.add_hline(
                    y=price,
                    line=dict(color="rgba(38,166,154,0.45)", width=1, dash="dash"),
                    annotation=dict(
                        text=f"지지 {price:,.0f} ({touches}회)",
                        font=dict(color="#26a69a", size=10),
                    ),
                    annotation_position="right",
                    row=1,
                    col=1,
                )

    # 매매 마커 (매매일지에서 호출 시)
    if trade_markers:
        for trade in trade_markers:
            try:
                dt = pd.to_datetime(trade["date"])
                if dt < df.index.min() or dt > df.index.max():
                    continue
                color = "#26a69a" if trade["action"] == "매수" else "#ef5350"
                symbol = "triangle-up" if trade["action"] == "매수" else "triangle-down"
                fig.add_trace(
                    go.Scatter(
                        x=[dt],
                        y=[trade["price"]],
                        mode="markers",
                        marker=dict(size=16, color=color, symbol=symbol, line=dict(color="white", width=1.5)),
                        name=f"{trade['action']} @ {trade['price']:,.0f}",
                        hovertext=f"{trade['action']} {trade.get('qty', '')}주<br>{trade.get('reason', '')}",
                        hoverinfo="text",
                    ),
                    row=1,
                    col=1,
                )
            except Exception:
                continue

    # 거래량
    vol_colors = [
        "#ef5350" if c >= o else "#1565c0" for c, o in zip(df["Close"], df["Open"])
    ]
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["Volume"],
            name="거래량",
            marker_color=vol_colors,
            opacity=0.8,
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # RSI
    if "RSI" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["RSI"],
                name="RSI",
                line=dict(color="#ff6b6b", width=1.5),
                showlegend=False,
            ),
            row=3,
            col=1,
        )
        fig.add_hline(y=70, line_dash="dash", line_color="rgba(239,83,80,0.5)", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="rgba(21,101,192,0.5)", row=3, col=1)
        fig.add_hrect(y0=70, y1=100, fillcolor="rgba(239,83,80,0.06)", row=3, col=1)
        fig.add_hrect(y0=0, y1=30, fillcolor="rgba(21,101,192,0.06)", row=3, col=1)

    # MACD
    if all(c in df.columns for c in ["MACD", "MACD_signal", "MACD_hist"]):
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["MACD"],
                name="MACD",
                line=dict(color="#29b6f6", width=1.5),
                showlegend=False,
            ),
            row=4,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["MACD_signal"],
                name="Signal",
                line=dict(color="#ff9800", width=1.5),
                showlegend=False,
            ),
            row=4,
            col=1,
        )
        hist_colors = ["#ef5350" if v >= 0 else "#1565c0" for v in df["MACD_hist"]]
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["MACD_hist"],
                name="Histogram",
                marker_color=hist_colors,
                opacity=0.7,
                showlegend=False,
            ),
            row=4,
            col=1,
        )

    fig.update_layout(
        height=850,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        plot_bgcolor="#131722",
        paper_bgcolor="#131722",
        font=dict(color="#d1d4dc", size=11),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    for i in range(1, 5):
        fig.update_xaxes(gridcolor="#1e2130", zerolinecolor="#1e2130", row=i, col=1)
        fig.update_yaxes(gridcolor="#1e2130", zerolinecolor="#1e2130", row=i, col=1)
    fig.update_yaxes(range=[0, 100], row=3, col=1)
    return fig


# =========================
# JSON 저장/로드 (관심종목, 매매일지)
# =========================
def load_watchlist() -> list[dict]:
    if not WATCHLIST_FILE.exists():
        return []
    try:
        return json.loads(WATCHLIST_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_watchlist(watchlist: list[dict]) -> None:
    WATCHLIST_FILE.write_text(
        json.dumps(watchlist, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def add_to_watchlist(ticker: str, name: str = "") -> bool:
    watchlist = load_watchlist()
    if any(w["ticker"] == ticker for w in watchlist):
        return False
    watchlist.append(
        {
            "ticker": ticker,
            "name": name or ticker,
            "added_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
    )
    save_watchlist(watchlist)
    return True


def remove_from_watchlist(ticker: str) -> None:
    watchlist = load_watchlist()
    watchlist = [w for w in watchlist if w["ticker"] != ticker]
    save_watchlist(watchlist)


def load_trades() -> list[dict]:
    if not TRADES_FILE.exists():
        return []
    try:
        return json.loads(TRADES_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_trades(trades: list[dict]) -> None:
    TRADES_FILE.write_text(
        json.dumps(trades, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def add_trade(trade: dict) -> None:
    trades = load_trades()
    trade["id"] = datetime.now().strftime("%Y%m%d%H%M%S%f")
    trade["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    trades.append(trade)
    save_trades(trades)


def delete_trade(trade_id: str) -> None:
    trades = load_trades()
    trades = [t for t in trades if t.get("id") != trade_id]
    save_trades(trades)


# =========================
# 특정 날짜 주가 조회 (매매일지/시뮬레이터용)
# =========================
@st.cache_data(ttl=3600)
def get_price_on_date(ticker: str, date_str: str) -> float | None:
    """해당 날짜(또는 가장 가까운 거래일)의 종가"""
    try:
        date = pd.to_datetime(date_str)
        start = (date - timedelta(days=7)).strftime("%Y-%m-%d")
        end = (date + timedelta(days=3)).strftime("%Y-%m-%d")
        df = yf.Ticker(ticker).history(start=start, end=end)
        if df.empty:
            return None
        if df.index.tzinfo is not None:
            df.index = df.index.tz_localize(None)
        # 지정일 이하의 가장 최근 거래일
        valid = df[df.index <= date]
        if valid.empty:
            return float(df["Close"].iloc[0])
        return float(valid["Close"].iloc[-1])
    except Exception:
        return None


# =========================
# AI 분석 (Claude)
# =========================
def _get_anthropic_client() -> anthropic.Anthropic | None:
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key or api_key == "your_api_key_here":
        return None
    try:
        return anthropic.Anthropic(api_key=api_key)
    except Exception:
        return None


def ai_analyze_chart(
    ticker: str,
    df: pd.DataFrame,
    signals: dict,
    info: dict,
    news: list,
    fundamentals: dict,
    support: list,
    resistance: list,
) -> str:
    """차트 종합 분석 (기본 페이지)"""
    client = _get_anthropic_client()
    if client is None:
        return _api_key_help()

    latest = df.iloc[-1]
    ccy = currency_name(ticker)

    price_changes = {}
    for label, n in [("1주", 5), ("1개월", 20), ("3개월", 60)]:
        if len(df) > n:
            pct = ((latest["Close"] / df.iloc[-n]["Close"]) - 1) * 100
            price_changes[label] = f"{pct:+.1f}%"

    news_text = "\n".join(f"- {n.get('title', '')}" for n in news[:5]) or "없음"

    sr_text = ""
    if resistance:
        sr_text += "저항선: " + ", ".join(f"{p:,.0f}({t}회)" for p, t in resistance[:3]) + "\n"
    if support:
        sr_text += "지지선: " + ", ".join(f"{p:,.0f}({t}회)" for p, t in support[:3])

    fund_text = "\n".join(f"- {k}: {v}" for k, v in fundamentals.items())

    prompt = f"""당신은 보수적인 장기 투자자를 돕는 주식 분석 전문가입니다.

투자자 프로필:
- 목표: 원금 보전 + 인플레이션 대항 수익 (연 5~10%)
- 레버리지 없음, 리스크 최소화
- 추세 추종 (최저/최고점 불필요)
- 분할 매수/매도 선호

종목: {ticker} ({info.get('longName', ticker)})
현재가: {latest['Close']:,.2f} {ccy}
가격 변화: {', '.join(f'{k} {v}' for k, v in price_changes.items())}

[기술적 지표]
- RSI ({signals.get('rsi', {}).get('value', 'N/A')}): {signals.get('rsi', {}).get('signal', 'N/A')}
- MACD: {signals.get('macd', {}).get('signal', 'N/A')}
- 볼린저밴드: {signals.get('bollinger', {}).get('signal', 'N/A')}
- 이동평균: {signals.get('ma', {}).get('signal', 'N/A')}
- 종합 점수: {signals.get('score', 0)} / 현재 판단: {signals.get('overall', 'N/A')}

[지지/저항선]
{sr_text or '계산 불가'}

[기본 재무 지표]
{fund_text}

[최근 뉴스]
{news_text}

아래 형식으로 분석해주세요 (한국어, 초보자 이해 쉽게):

## 현재 상황 한줄 요약

## 차트에서 주목할 포인트 (2~3가지)

## 밸류에이션 코멘트 (PER/PBR/배당 등)

## 지금의 위험 요소

## 매수 타이밍 제안 (구체적 조건)

## 매도 타이밍 제안 (구체적 조건)

## 보수적 투자자에게 한마디

⚠️ 투자 확정 추천 금지. 객관적 분석만."""

    try:
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2200,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text
    except Exception as e:
        return f"분석 오류: {e}"


def ai_analyze_trades(trades: list[dict]) -> str:
    """매매일지 전체 패턴 분석"""
    client = _get_anthropic_client()
    if client is None:
        return _api_key_help()

    if not trades:
        return "분석할 매매 기록이 없습니다. 매매 일지를 먼저 추가해주세요."

    # 매매 기록 텍스트화 (차트 맥락 포함)
    lines = []
    for t in trades[-30:]:  # 최근 30건
        ticker = t.get("ticker", "")
        date = t.get("date", "")
        action = t.get("action", "")
        price = t.get("price", 0)
        qty = t.get("qty", 0)
        reason = t.get("reason", "(이유 없음)")
        ctx = t.get("context", {})
        ctx_str = ""
        if ctx:
            ctx_str = (
                f"  [당시 차트: RSI={ctx.get('rsi', 'N/A')}, "
                f"MA20>MA60={ctx.get('ma_trend', 'N/A')}, "
                f"신호점수={ctx.get('score', 'N/A')}]"
            )
        lines.append(f"- {date} {ticker} {action} {qty}주 @ {price:,.0f}\n  이유: {reason}{ctx_str}")
    trades_text = "\n".join(lines)

    prompt = f"""당신은 개인 투자자의 매매 패턴을 분석해주는 코치입니다.

투자자 프로필:
- 목표: 원금 보전 + 인플레이션 대항 (연 5~10%)
- 레버리지 없음, 리스크 최소화, 추세 추종
- 초보 투자자

아래는 투자자가 직접 기록한 매매 일지입니다. 각 거래에는 투자자의 이유와 당시 기술적 지표 스냅샷이 함께 있습니다.

[매매 기록]
{trades_text}

다음 항목으로 코칭해주세요 (한국어, 따뜻하지만 솔직하게):

## 전반적 평가
(이 투자자의 매매 스타일을 한두 문장으로)

## 잘하고 있는 점 (2~3가지)
(구체적 거래 예시와 함께)

## 고칠 점 또는 주의할 패턴 (2~3가지)
(예: "RSI 과매수 구간에서 매수" / "손절 타이밍을 놓침" / "이유가 감정적" 등 구체적으로)

## 매수 이유 분석
(투자자가 적은 이유들이 차트 상황과 일치했는지, 근거가 탄탄한지)

## 매도 이유 분석
(차익실현/손절이 적절했는지)

## 다음 매매 전에 체크할 리스트
(구체적 3~5개 질문)

## 격려의 한마디

⚠️ 과거 거래의 수익률을 계산하지 마세요. 과거 시점의 결정 근거와 패턴에만 집중하세요."""

    try:
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2500,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text
    except Exception as e:
        return f"분석 오류: {e}"


def ai_analyze_single_trade(trade: dict, context_df: pd.DataFrame | None) -> str:
    """특정 거래 하나에 대한 심층 분석"""
    client = _get_anthropic_client()
    if client is None:
        return _api_key_help()

    ctx_snapshot = ""
    if context_df is not None and not context_df.empty:
        df = calculate_indicators(context_df.copy())
        sigs = analyze_signals(df)
        ctx_snapshot = f"""
[당시 기술적 지표]
- RSI: {sigs.get('rsi', {}).get('value', 'N/A')} - {sigs.get('rsi', {}).get('signal', '')}
- MACD: {sigs.get('macd', {}).get('signal', '')}
- 볼린저밴드: {sigs.get('bollinger', {}).get('signal', '')}
- 이동평균: {sigs.get('ma', {}).get('signal', '')}
- 종합 신호: {sigs.get('overall', '')} (점수 {sigs.get('score', 0)})
"""

    prompt = f"""아래는 투자자가 기록한 하나의 매매 건입니다. 이 결정이 당시 차트 상황에 비춰봤을 때 타당했는지 분석해주세요.

[거래]
- 종목: {trade.get('ticker', '')}
- 날짜: {trade.get('date', '')}
- 행동: {trade.get('action', '')}
- 가격: {trade.get('price', 0):,.2f}
- 수량: {trade.get('qty', 0)}
- 투자자 이유: {trade.get('reason', '(없음)')}

{ctx_snapshot}

다음을 짧고 명료하게 한국어로:

## 결정 요약 (한 줄)
## 이유의 타당성 (차트와 일치/불일치)
## 당시 더 좋은 대안이 있었는지
## 이번 경험에서 배울 점 (1~2개)

⚠️ 사후편향(hindsight bias) 금지. 그 시점에 알 수 있던 정보로만 판단."""

    try:
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text
    except Exception as e:
        return f"분석 오류: {e}"


def _api_key_help() -> str:
    return (
        "⚠️ **API 키가 설정되지 않았습니다.**\n\n"
        "1. 프로젝트 폴더의 `.env` 파일을 메모장으로 열기\n"
        "   (없으면 `.env.example`을 `.env`로 복사)\n"
        "2. `ANTHROPIC_API_KEY=` 뒤에 발급받은 키 붙여넣기\n"
        "3. 저장 후 앱 재시작\n\n"
        "[API 키 발급 → console.anthropic.com](https://console.anthropic.com)"
    )
