"""
한국투자증권 Open API 연동
- 계좌 잔고 조회
- 보유 종목 실시간 현황
"""

from __future__ import annotations

import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

KIS_APP_KEY = os.getenv("KIS_APP_KEY", "")
KIS_APP_SECRET = os.getenv("KIS_APP_SECRET", "")
KIS_ACCOUNT_NO = os.getenv("KIS_ACCOUNT_NO", "")  # 8자리 계좌번호

BASE_URL = "https://openapi.koreainvestment.com:9443"


def is_configured() -> bool:
    return bool(KIS_APP_KEY and KIS_APP_SECRET and KIS_ACCOUNT_NO)


@st.cache_data(ttl=3600)
def get_access_token() -> str | None:
    """OAuth 토큰 발급 (24시간 유효, 1시간 캐시)"""
    if not is_configured():
        return None
    try:
        resp = requests.post(
            f"{BASE_URL}/oauth2/tokenP",
            json={
                "grant_type": "client_credentials",
                "appkey": KIS_APP_KEY,
                "appsecret": KIS_APP_SECRET,
            },
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json().get("access_token")
    except Exception:
        return None


@st.cache_data(ttl=60)
def get_portfolio() -> tuple[list[dict], dict]:
    """
    주식 잔고 조회.
    반환: (보유종목 리스트, 계좌 요약)
    """
    token = get_access_token()
    if not token:
        return [], {}

    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {token}",
        "appkey": KIS_APP_KEY,
        "appsecret": KIS_APP_SECRET,
        "tr_id": "TTTC8434R",  # 실전계좌 잔고조회
        "custtype": "P",
    }
    params = {
        "CANO": KIS_ACCOUNT_NO,
        "ACNT_PRDT_CD": "01",
        "AFHR_FLPR_YN": "N",
        "OFL_YN": "N",
        "INQR_DVSN": "02",
        "UNPR_DVSN": "01",
        "FUND_STTL_ICLD_YN": "N",
        "FNCG_AMT_AUTO_RDPT_YN": "N",
        "PRCS_DVSN": "00",
        "CTX_AREA_FK100": "",
        "CTX_AREA_NK100": "",
    }

    try:
        resp = requests.get(
            f"{BASE_URL}/uapi/domestic-stock/v1/trading/inquire-balance",
            headers=headers,
            params=params,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("rt_cd") != "0":
            return [], {"error": data.get("msg1", "알 수 없는 오류")}

        def _float(v, default=0.0) -> float:
            try:
                return float(v or default)
            except (ValueError, TypeError):
                return default

        def _int(v, default=0) -> int:
            try:
                return int(v or default)
            except (ValueError, TypeError):
                return default

        holdings = []
        for item in data.get("output1", []):
            qty = _int(item.get("hldg_qty"))
            if qty <= 0:
                continue
            code = item.get("pdno", "")
            holdings.append({
                "name": item.get("prdt_name", code),
                "ticker": f"{code}.KS",
                "code": code,
                "qty": qty,
                "avg_price": _float(item.get("pchs_avg_pric")),
                "current_price": _float(item.get("prpr")),
                "eval_amount": _float(item.get("evlu_amt")),
                "purchase_amount": _float(item.get("pchs_amt")),
                "profit_loss": _float(item.get("evlu_pfls_amt")),
                "profit_loss_pct": _float(item.get("evlu_pfls_rt")),
            })

        # 계좌 요약 (output2)
        summary = {}
        out2 = data.get("output2", [])
        if out2:
            s = out2[0]
            summary = {
                "total_eval": _float(s.get("tot_evlu_amt")),
                "total_purchase": _float(s.get("pchs_amt_smtl_amt")),
                "total_profit_loss": _float(s.get("evlu_pfls_smtl_amt")),
                "profit_loss_pct": _float(s.get("asst_icdc_erng_rt")),
                "cash": _float(s.get("dnca_tot_amt")),
            }

        return holdings, summary

    except requests.exceptions.ConnectionError:
        return [], {"error": "연결 실패 — 인터넷 연결을 확인하세요."}
    except requests.exceptions.Timeout:
        return [], {"error": "응답 시간 초과 — 잠시 후 다시 시도하세요."}
    except Exception as e:
        return [], {"error": str(e)}
