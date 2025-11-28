# tools.py
"""
Tool functions used by the solver.

These are the "hands" that the LLM (planner) can decide to use:
- Web scraping (simple HTTP)
- JS rendering (Playwright)
- API calls
- Basic data loading & aggregation (pandas)
- Placeholder chart generation
"""

from __future__ import annotations

import io
import time
from typing import Any, Dict, Optional

import requests

import pandas as pd  # type: ignore
from bs4 import BeautifulSoup  # type: ignore

from playwright.async_api import async_playwright  # type: ignore


# ------------------ Web & scraping tools ------------------


def fetch_html(url: str, timeout: int = 20) -> str:
    """Simple HTTP GET; no JS execution."""
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.text


async def render_with_js(url: str, wait_until: str = "networkidle") -> str:
    """
    Use Playwright to load a page and execute JavaScript, then return HTML.

    Requires:
        pip install playwright
        python -m playwright install
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(url, wait_until=wait_until)
        content = await page.content()
        await browser.close()
    return content


# ------------------ API tools ------------------


def fetch_api_json(
    url: str,
    method: str = "GET",
    headers: Optional[Dict[str, str]] = None,
    payload: Optional[Dict[str, Any]] = None,
    timeout: int = 20,
) -> Dict[str, Any]:
    """Generic JSON API helper."""
    method = method.upper()
    if method == "GET":
        resp = requests.get(url, headers=headers, timeout=timeout)
    else:
        resp = requests.request(method, url, headers=headers, json=payload, timeout=timeout)

    resp.raise_for_status()
    return resp.json()  # type: ignore[return-value]


# ------------------ Data tools (pandas) ------------------


def load_csv_from_url(url: str) -> pd.DataFrame:
    """Load CSV from a URL into a DataFrame."""
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return pd.read_csv(io.StringIO(resp.text))


def basic_groupby_sum(df: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    """Example aggregation: group by one column and sum another."""
    grouped = df.groupby(group_col, as_index=False)[value_col].sum()
    return grouped.sort_values(value_col, ascending=False)


# ------------------ Visualization stub ------------------


def generate_chart_placeholder(df: pd.DataFrame, description: str) -> str:
    """
    Stub: in a real system, create a chart and return a Base64 URI.

    Here we just return a text marker so the pipeline doesn't break.
    """
    return f"[CHART_PLACEHOLDER for {description} with shape {df.shape}]"