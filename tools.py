# tools.py
"""
Tool functions used by the solver.

These are the "hands" that the LLM (planner) can decide to use:
- Web scraping (simple HTTP)
- JS rendering (Playwright)
- API calls
- Basic data loading & aggregation (pandas)
- Placeholder chart generation
- Image/audio/CSV helpers for the project2 challenge
"""

from __future__ import annotations

import io
import time
from typing import Any, Dict, Optional
from urllib.parse import urljoin

import requests
import pandas as pd  # type: ignore
from bs4 import BeautifulSoup  # type: ignore

# Playwright used only in render_with_js (async)
from playwright.async_api import async_playwright  # type: ignore

# Pillow for image helpers (optional)
try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = None  # type: ignore

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


# ------------------ Visualization helper (placeholder) ------------------


def generate_chart_placeholder(plan: dict) -> bytes:
    """
    Create a simple PNG placeholder for visualization plans.
    Returns raw PNG bytes.

    Accepts the plan dict used by solver.solve_visualization_quiz.
    """
    # If Pillow not available, return plain bytes of text
    if Image is None:
        return f"[CHART_PLACEHOLDER: {plan}]".encode("utf-8")

    w, h = 640, 360
    img = Image.new("RGB", (w, h), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    lines = []
    lines.append(f"Type: {plan.get('type')}")
    lines.append(f"X: {plan.get('x')}, Y: {plan.get('y')}")
    title = plan.get("title", "")
    if title:
        lines.append(f"Title: {title}")
    txt = "\n".join(lines)

    draw.multiline_text((12, 12), txt, fill=(0, 0, 0), font=font, spacing=4)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ------------------ Extra helpers for the challenge pages ------------------


def dominant_hex_from_image_url(url: str, timeout: int = 30) -> str:
    """
    Download an image and return the dominant (most frequent) pixel color as a hex string.
    Falls back to average color if necessary.
    """
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    data = r.content
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        # fallback: try basic average from raw bytes (not ideal)
        # return an empty string to indicate failure
        return ""

    # Downscale to speed up calculation
    try:
        img = img.resize((200, 200))
    except Exception:
        pass

    pixels = list(img.getdata())
    from collections import Counter

    most_common = Counter(pixels).most_common(1)
    if most_common:
        r_, g_, b_ = most_common[0][0]
        return '#{:02x}{:02x}{:02x}'.format(r_, g_, b_)

    # average fallback
    if pixels:
        r_avg = sum(p[0] for p in pixels) // len(pixels)
        g_avg = sum(p[1] for p in pixels) // len(pixels)
        b_avg = sum(p[2] for p in pixels) // len(pixels)
        return '#{:02x}{:02x}{:02x}'.format(r_avg, g_avg, b_avg)

    return ""


def csv_to_json_array(csv_url: str, timeout: int = 30) -> str:
    """Load CSV from url and return a JSON array string (list of records)."""
    r = requests.get(csv_url, timeout=timeout)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    # basic normalization
    df.columns = [c.strip() for c in df.columns]
    for c in df.select_dtypes(include=['object']).columns:
        df[c] = df[c].astype(str).str.strip()
    arr = df.to_dict(orient='records')
    import json
    return json.dumps(arr, ensure_ascii=False)


def transcribe_audio_from_url(url: str, timeout: int = 30) -> str:
    """
    Try to transcribe audio from a URL.
    Preferred path: whisper (if installed).
    Fallback: pydub + speech_recognition's Google recognizer (requires internet).
    Returns text or empty string on failure.
    """
    import tempfile
    import os
    import json

    # Try whisper first
    try:
        import whisper  # type: ignore
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tmp.write(r.content)
        tmp.close()
        model = whisper.load_model("small")
        res = model.transcribe(tmp.name)
        os.unlink(tmp.name)
        return (res.get("text") or "").strip()
    except Exception:
        # fallback
        pass

    # fallback to pydub + speech_recognition
    try:
        from pydub import AudioSegment  # type: ignore
        import speech_recognition as sr  # type: ignore
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        audio = AudioSegment.from_file(io.BytesIO(r.content))
        audio.export(tmp.name, format="wav")
        rec = sr.Recognizer()
        with sr.AudioFile(tmp.name) as source:
            audio_data = rec.record(source)
        os.unlink(tmp.name)
        try:
            text = rec.recognize_google(audio_data)
            return text.strip()
        except Exception:
            return ""
    except Exception:
        return ""


def count_md_files_github_tree(repo_url: str, path_prefix: str, timeout: int = 30) -> int:
    """
    Count .md files under path_prefix in a GitHub repo using API.
    repo_url example: https://github.com/owner/repo
    """
    import re
    m = re.match(r"https?://github.com/([^/]+)/([^/]+)", repo_url)
    if not m:
        return 0
    owner = m.group(1); repo = m.group(2)
    api_repo = f"https://api.github.com/repos/{owner}/{repo}"
    r = requests.get(api_repo, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    branch = data.get("default_branch", "main")
    tree_api = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
    tr = requests.get(tree_api, timeout=timeout)
    tr.raise_for_status()
    tree = tr.json().get("tree", [])
    # normalize prefix (strip leading/trailing slashes)
    prefix = path_prefix.strip("/")
    count = 0
    for node in tree:
        path = node.get("path", "")
        if not prefix:
            if path.lower().endswith(".md"):
                count += 1
        else:
            if path.startswith(prefix) and path.lower().endswith(".md"):
                count += 1
    return count