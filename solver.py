from __future__ import annotations

import time
import re
import json
import logging
import ast
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, parse_qs, urljoin

import requests
from bs4 import BeautifulSoup  # type: ignore

from tools import (
    fetch_html,
    render_with_js,
    fetch_api_json,
    load_csv_from_url,
    generate_chart_placeholder,
)
from llm_client import call_llm

# ==========================================================
# LOGGING
# ==========================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("solver")


# ==========================================================
# HELPER: SANITIZE & PARSE LLM ANSWERS
# ==========================================================
def sanitize_answer(ans: Optional[str]) -> str:
    """Remove chain-of-thought and prefixes, return first meaningful line."""
    if not ans:
        return ""
    s = str(ans)

    # remove <think>...</think> and similar reasoning tags
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL | re.IGNORECASE)
    s = re.sub(r"<.*?reasoning.*?>.*?</.*?>", "", s, flags=re.DOTALL | re.IGNORECASE)

    s = s.strip()
    lines = [l.strip() for l in s.splitlines() if l.strip()]
    if not lines:
        return ""

    first = lines[0]

    # strip common prefixes
    first = re.sub(
        r'(?i)^(final answer[:\-\s]*|answer[:\-\s]*|the answer[:\-\s]*|respond (only )?with[:\-\s]*)',
        '',
        first,
    ).strip()

    first = first.rstrip(". \t\n\r")
    first = first.strip().strip('"\'')

    return first


def parse_possible_structured_answer(s: str):
    """
    Try to interpret 's' as JSON or Python literal (single-quoted dict).
    Returns Python object on success, else None.
    """
    if not s:
        return None
    s = s.strip()

    # JSON attempt
    try:
        return json.loads(s)
    except Exception:
        pass

    # Python literal (e.g. "{'label': 'Yes', 'answer': True}")
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, (dict, list, str, int, float, bool)):
            return obj
    except Exception:
        pass

    # Try to extract dict-like substring
    m = re.search(r"(\{.*\})", s, flags=re.DOTALL)
    if m:
        inner = m.group(1)
        try:
            return json.loads(inner)
        except Exception:
            try:
                return ast.literal_eval(inner)
            except Exception:
                pass

    return None


def coerce_answer_type(value: Any) -> Any:
    """
    Accepts string or parsed object.
    - If dict/list/bool/int/float, return as-is.
    - If string: try structured parse, then bool/int/float, else return string.
    """
    if value is None:
        return None

    if isinstance(value, (dict, list, bool, int, float)):
        return value

    s = str(value).strip()

    parsed = parse_possible_structured_answer(s)
    if parsed is not None:
        if isinstance(parsed, dict):
            for k in ("answer", "value", "result", "final"):
                if k in parsed:
                    return parsed[k]
        return parsed

    low = s.lower()
    if low in {"true", "yes"}:
        return True
    if low in {"false", "no"}:
        return False

    if re.fullmatch(r"[-+]?\d+", s):
        try:
            return int(s)
        except Exception:
            pass

    if re.fullmatch(r"[-+]?\d*\.\d+(e[-+]?\d+)?", s, flags=re.I) or re.fullmatch(
        r"[-+]?\d+(\.\d*)?[eE][-+]?\d+",
        s,
    ):
        try:
            return float(s)
        except Exception:
            pass

    return s


# ==========================================================
# MAIN SOLVER LOOP
# ==========================================================
async def solve_quiz_chain(start_url: str, email: str, secret: str) -> Dict[str, Any]:
    current_url: Optional[str] = start_url
    solved = 0
    history: List[Dict[str, Any]] = []

    start_time = time.time()
    MAX_TIME = 170
    MAX_QUIZZES = 20

    while current_url:
        if time.time() - start_time > MAX_TIME:
            return {"status": "timeout", "solved": solved, "history": history}

        if solved >= MAX_QUIZZES:
            return {"status": "max_limit_reached", "solved": solved, "history": history}

        logger.info(f"===== Solving quiz: {current_url} =====")
        html = fetch_html(current_url)
        if not html:
            history.append({"url": current_url, "error": "failed_fetch_html"})
            break

        quiz_info = parse_quiz_page(html, current_url)

        if not quiz_info.get("email"):
            quiz_info["email"] = email

        logger.info("Parsed quiz_info keys=" + str(list(quiz_info.keys())))

        try:
            raw_answer = await compute_answer(quiz_info)
            logger.info(f"Raw computed answer: {raw_answer}")
        except Exception as e:
            logger.exception("Error during compute_answer")
            history.append({"url": current_url, "error": str(e)})
            break

        # Prepare final answer for submission (sanitize + structured parsing)
        if isinstance(raw_answer, (dict, list, bool, int, float)):
            final_answer = raw_answer
        else:
            cleaned = sanitize_answer(str(raw_answer))
            parsed = parse_possible_structured_answer(cleaned)
            if parsed is not None:
                if isinstance(parsed, dict):
                    for k in ("answer", "value", "result", "final"):
                        if k in parsed:
                            final_answer = parsed[k]
                            break
                    else:
                        final_answer = parsed
                else:
                    final_answer = parsed
            else:
                final_answer = coerce_answer_type(cleaned)

        logger.info(f"Final answer to submit: {final_answer!r}")

        resp = submit_answer(
            email=email,
            secret=secret,
            quiz_url=current_url,
            answer=final_answer,
            submission_url=quiz_info["submission_url"],
        )

        history.append({
            "url": current_url,
            "answer": final_answer,
            "correct": resp.get("correct"),
            "next_url": resp.get("url"),
            "reason": resp.get("reason", "")
        })

        if resp.get("correct"):
            solved += 1

        next_url = resp.get("url")
        if not next_url:
            break

        current_url = next_url

    return {"status": "completed", "solved": solved, "history": history}


# ==========================================================
# PAGE PARSING
# ==========================================================
def parse_quiz_page(html: str, url: str) -> Dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")

    question_el = (
        soup.find("div", {"id": "question"})
        or soup.find("div", {"class": "question"})
        or soup.find("h1")
    )
    question_text = question_el.get_text(strip=True) if question_el else ""

    scripts = soup.find_all("script")
    has_js_module = any(
        (s.get("type") == "module") or (s.get("src") or "").endswith(".js")
        for s in scripts
    )

    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    email = qs.get("email", [None])[0]

    submission_url: Optional[str] = None

    # Case 1: <form action="...">
    form = soup.find("form")
    if form and form.get("action"):
        submission_url = urljoin(url, form.get("action"))

    # Case 2: links/buttons containing /submit
    if not submission_url:
        for a in soup.find_all("a", href=True):
            href = a.get("href", "")
            if "/submit" in href:
                submission_url = urljoin(url, href)
                break

    if not submission_url:
        for b in soup.find_all(["button", "input"], {"formaction": True}):
            fa = b.get("formaction")
            if fa and "/submit" in fa:
                submission_url = urljoin(url, fa)
                break

    # Case 3: scripts containing an absolute /submit
    if not submission_url:
        for tag in soup.find_all(["script", "div", "span"]):
            txt = tag.string or tag.get_text(" ", strip=True) or ""
            if "/submit" in txt:
                m = re.search(r"https?://[^\"' ]+/submit", txt)
                if m:
                    submission_url = m.group(0)
                    break

    # Case 4: fallback
    if not submission_url:
        submission_url = f"{parsed.scheme}://{parsed.netloc}/submit"

    return {
        "url": url,
        "email": email,
        "question_text": question_text,
        "has_js_module": has_js_module,
        "submission_url": submission_url,
    }


# ==========================================================
# TYPE DETECTION
# ==========================================================
def detect_quiz_type(quiz_info: Dict[str, Any]) -> str:
    url = (quiz_info.get("url") or "").lower()
    text = (quiz_info.get("question_text") or "").lower()

    # JS-heavy
    if quiz_info.get("has_js_module") or "scrape" in url:
        return "js_scrape"

    # API hint
    if "api" in url or "endpoint" in text or "fetch" in text:
        return "api"

    # CSV / dataset
    if ".csv" in text or "dataset" in text or "data" in text:
        return "data_csv"

    # Visualization
    if any(k in text for k in ["chart", "plot", "visualize", "graph", "bar", "line chart"]):
        return "visualization"

    # Fallback: classifier
    prompt = f"""
Classify this question into one category:

- js_scrape
- api
- data_csv
- visualization
- text_reasoning

QUESTION:
{text}

URL:
{url}

Respond with ONLY the label.
"""
    raw = call_llm(prompt.strip(), system="Classifier")
    label = sanitize_answer(raw).lower()

    if label not in {"js_scrape", "api", "data_csv", "visualization", "text_reasoning"}:
        return "text_reasoning"

    return label


# ==========================================================
# DISPATCHER
# ==========================================================
async def compute_answer(quiz_info: Dict[str, Any]) -> Any:
    quiz_type = detect_quiz_type(quiz_info)
    logger.info(f"Detected type: {quiz_type} for URL {quiz_info.get('url')}")

    if quiz_type == "js_scrape":
        return await solve_js_scrape(quiz_info)
    if quiz_type == "api":
        return await solve_api_quiz(quiz_info)
    if quiz_type == "data_csv":
        return await solve_data_quiz(quiz_info)
    if quiz_type == "visualization":
        return await solve_visualization_quiz(quiz_info)
    return await solve_text_reasoning_quiz(quiz_info)


# ==========================================================
# JS SCRAPE HANDLER
# ==========================================================
async def solve_js_scrape(quiz_info: Dict[str, Any]) -> Any:
    url = quiz_info["url"]
    logger.info(f"JS scrape for: {url}")

    try:
        rendered_html = await render_with_js(url)
        soup = BeautifulSoup(rendered_html, "html.parser")
    except Exception as e:
        logger.info(f"render_with_js failed ({e}), falling back to plain HTML")
        html = fetch_html(url)
        soup = BeautifulSoup(html, "html.parser")

    q_div = soup.find("div", {"id": "question"})
    text = q_div.get_text(" ", strip=True) if q_div else soup.get_text(" ", strip=True)

    if not text.strip():
        raise ValueError("JS-rendered page has no visible text")

    # direct secret/code pattern
    m = re.search(r"(secret|code)\s*[:\- ]*\s*([0-9]{2,})", text, flags=re.IGNORECASE)
    if m:
        return sanitize_answer(m.group(2))

    # JSON-like embedded
    mjson = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if mjson:
        try:
            jsdata = json.loads(mjson.group(0))
            if isinstance(jsdata, dict):
                for k in ("secret", "code", "answer", "value"):
                    if k in jsdata:
                        return sanitize_answer(str(jsdata[k]))
        except Exception:
            pass

    # fallback to LLM extraction
    prompt = f"""
Extract the final answer from the web page content below.
Return ONLY the final answer.

CONTENT:
{text}
"""
    ans = call_llm(prompt.strip(), system="Answer extractor")
    return sanitize_answer(ans)


# ==========================================================
# API HANDLER
# ==========================================================
async def solve_api_quiz(quiz_info: Dict[str, Any]) -> Any:
    text = quiz_info.get("question_text", "")

    prompt = f"""
You are an API planner.

Read the question and produce a JSON instruction plan with these fields:

{{
  "endpoint": "API URL",
  "method": "GET or POST",
  "params": {{ ... }},
  "extract": "key to extract from JSON, if any"
}}

QUESTION:
{text}

Respond ONLY with valid JSON.
"""
    raw = call_llm(prompt.strip(), system="API Planner")
    plan_str = sanitize_answer(raw)

    try:
        plan = json.loads(plan_str)
    except Exception:
        raise ValueError(f"LLM returned invalid API JSON: {plan_str}")

    endpoint = plan.get("endpoint")
    method = (plan.get("method") or "GET").upper()
    params = plan.get("params") or {}

    if not endpoint:
        raise ValueError("API plan missing endpoint")

    data = fetch_api_json(endpoint, method=method, params=params)

    extract_key = plan.get("extract")
    if extract_key and isinstance(data, dict) and extract_key in data:
        return sanitize_answer(str(data[extract_key]))

    return sanitize_answer(str(data))


# ==========================================================
# CSV / DATA QUIZ HANDLER
# ==========================================================
async def solve_data_quiz(quiz_info: Dict[str, Any]) -> Any:
    text = (quiz_info.get("question_text") or "").strip()
    if not text:
        raise ValueError("Empty question text for data quiz.")

    # find CSV URL in text or page HTML
    match = re.search(r"(https?://\S+?\.csv)(?:\s|$)", text, re.IGNORECASE)
    if not match:
        page_html = fetch_html(quiz_info.get("url", ""))
        if page_html:
            match = re.search(r"(https?://\S+?\.csv)(?:\s|<|\"|')", page_html, re.IGNORECASE)
    if not match:
        raise ValueError("No CSV URL found in question.")
    csv_url = match.group(1)

    # load CSV
    try:
        df = load_csv_from_url(csv_url)
    except Exception as e:
        raise ValueError(f"Failed to load CSV: {e}")

    if df is None or df.empty:
        raise ValueError("Loaded CSV is empty or unreadable.")

    from pandas.api.types import is_numeric_dtype

    def _pick_numeric_column():
        for c in df.columns:
            try:
                if is_numeric_dtype(df[c]):
                    return c
            except Exception:
                continue
        return None

    lower = text.lower()

    # groupby pattern
    gb_match = re.search(
        r"(?:group by|groupby)\s+([A-Za-z0-9_ \-]+).*?(?:sum|total|aggregate)\s+([A-Za-z0-9_ \-]+)",
        lower,
    )
    if gb_match:
        col_group = gb_match.group(1).strip()
        col_value = gb_match.group(2).strip()

        def _map_col(name):
            for c in df.columns:
                if c.lower() == name.lower() or name.lower() in c.lower():
                    return c
            return None

        cg = _map_col(col_group)
        cv = _map_col(col_value)
        if cg and cv:
            try:
                res = df.groupby(cg)[cv].sum().to_dict()
                return sanitize_answer(json.dumps(res))
            except Exception as e:
                raise ValueError(f"groupby failed: {e}")

    # sum
    sum_match = re.search(r"(?:sum|total|add|sum of)\s+([A-Za-z0-9_ \-]+)", lower)
    if sum_match:
        col_name = sum_match.group(1).strip()
        col = None
        for c in df.columns:
            if c.lower() == col_name.lower() or col_name.lower() in c.lower():
                col = c
                break
        if not col:
            col = _pick_numeric_column()
        if not col:
            raise ValueError("No numeric column available for sum.")
        try:
            val = df[col].sum()
            return coerce_answer_type(str(val))
        except Exception as e:
            raise ValueError(f"sum failed: {e}")

    # mean
    if "mean" in lower or "average" in lower:
        mean_match = re.search(r"(?:mean|average|avg)\s+of\s+([A-Za-z0-9_ \-]+)", lower)
        col = None
        if mean_match:
            name = mean_match.group(1).strip()
            for c in df.columns:
                if c.lower() == name.lower() or name.lower() in c.lower():
                    col = c
                    break
        if not col:
            col = _pick_numeric_column()
        if not col:
            raise ValueError("No numeric column available for mean.")
        try:
            val = df[col].mean()
            return coerce_answer_type(str(val))
        except Exception as e:
            raise ValueError(f"mean failed: {e}")

    # median
    if "median" in lower:
        med_match = re.search(r"median\s+of\s+([A-Za-z0-9_ \-]+)", lower)
        col = None
        if med_match:
            name = med_match.group(1).strip()
            for c in df.columns:
                if c.lower() == name.lower() or name.lower() in c.lower():
                    col = c
                    break
        if not col:
            col = _pick_numeric_column()
        if not col:
            raise ValueError("No numeric column available for median.")
        try:
            val = df[col].median()
            return coerce_answer_type(str(val))
        except Exception as e:
            raise ValueError(f"median failed: {e}")

    # count / distinct count
    if re.search(r"\b(count|how many|number of|rows)\b", lower):
        distinct_match = re.search(r"(?:count distinct|distinct count)\s+([A-Za-z0-9_ \-]+)", lower)
        if distinct_match:
            name = distinct_match.group(1).strip()
            col = None
            for c in df.columns:
                if c.lower() == name.lower() or name.lower() in c.lower():
                    col = c
                    break
            if not col:
                raise ValueError("Requested distinct count but column not found.")
            try:
                val = df[col].nunique()
                return coerce_answer_type(str(val))
            except Exception as e:
                raise ValueError(f"distinct count failed: {e}")
        return coerce_answer_type(str(len(df)))

    # max
    if "max" in lower or "maximum" in lower:
        m = re.search(r"(?:max|maximum)\s+of\s+([A-Za-z0-9_ \-]+)", lower)
        col = None
        if m:
            name = m.group(1).strip()
            for c in df.columns:
                if c.lower() == name.lower() or name.lower() in c.lower():
                    col = c
                    break
        if not col:
            col = _pick_numeric_column()
        if not col:
            raise ValueError("No numeric column for max.")
        try:
            val = df[col].max()
            return coerce_answer_type(str(val))
        except Exception as e:
            raise ValueError(f"max failed: {e}")

    # min
    if "min" in lower or "minimum" in lower:
        m = re.search(r"(?:min|minimum)\s+of\s+([A-Za-z0-9_ \-]+)", lower)
        col = None
        if m:
            name = m.group(1).strip()
            for c in df.columns:
                if c.lower() == name.lower() or name.lower() in c.lower():
                    col = c
                    break
        if not col:
            col = _pick_numeric_column()
        if not col:
            raise ValueError("No numeric column for min.")
        try:
            val = df[col].min()
            return coerce_answer_type(str(val))
        except Exception as e:
            raise ValueError(f"min failed: {e}")

    # Fallback: LLM JSON plan
    llm_prompt = f"""
You are a data assistant. CSV: {csv_url}
Question: {text}

Return a tiny JSON object (ONLY JSON) describing what to compute, for example:
{{"operation":"sum", "column":"Sales", "filter": "Region=='APAC'"}}
Allowed operations: sum, mean, median, count, distinct_count, max, min, groupby_sum.
For groupby_sum use keys: "groupby":"ColumnA", "agg":"ColumnB".
"""
    raw = call_llm(llm_prompt.strip(), system="Data planner (json)")
    cleaned = sanitize_answer(raw)
    try:
        plan = json.loads(cleaned)
    except Exception:
        raise ValueError("LLM returned unparseable plan for CSV. Plan text: " + cleaned)

    op = plan.get("operation")
    try:
        if op == "sum":
            col = plan.get("column") or _pick_numeric_column()
            if col not in df.columns:
                for c in df.columns:
                    if c.lower() == col.lower() or (col and col.lower() in c.lower()):
                        col = c
                        break
            if col is None:
                raise ValueError("No column found for sum")
            return coerce_answer_type(str(df[col].sum()))

        if op == "mean":
            col = plan.get("column") or _pick_numeric_column()
            return coerce_answer_type(str(df[col].mean()))

        if op == "median":
            col = plan.get("column") or _pick_numeric_column()
            return coerce_answer_type(str(df[col].median()))

        if op == "count":
            if plan.get("column"):
                col = plan["column"]
                return coerce_answer_type(str(int(df[col].count())))
            return coerce_answer_type(str(len(df)))

        if op == "distinct_count":
            col = plan.get("column")
            return coerce_answer_type(str(int(df[col].nunique())))

        if op == "max":
            col = plan.get("column") or _pick_numeric_column()
            return coerce_answer_type(str(df[col].max()))

        if op == "min":
            col = plan.get("column") or _pick_numeric_column()
            return coerce_answer_type(str(df[col].min()))

        if op == "groupby_sum":
            g = plan.get("groupby")
            agg = plan.get("agg")
            if not g or not agg:
                raise ValueError("groupby_sum requires 'groupby' and 'agg'")
            res = df.groupby(g)[agg].sum().to_dict()
            return sanitize_answer(json.dumps(res))

        raise ValueError(f"Unknown operation requested by plan: {op}")
    except Exception as e:
        raise ValueError(f"Failed to execute plan: {e}")


# ==========================================================
# VISUALIZATION HANDLER
# ==========================================================
async def solve_visualization_quiz(quiz_info: Dict[str, Any]) -> Any:
    text = quiz_info.get("question_text", "")

    prompt = f"""
You are a visualization assistant. Question: {text}

Return ONLY a JSON object describing the visualization plan:
{{"type":"bar|line|scatter|pie","x":"ColumnName","y":"ColumnName","title":"...","want_image":true|false}}
Respond with valid JSON only.
"""
    raw = call_llm(prompt.strip(), system="Visualization assistant (json)")
    cleaned = sanitize_answer(raw)

    try:
        plan = json.loads(cleaned)
    except Exception:
        return "[VISUALIZATION_PLAN] " + cleaned

    want_image = bool(plan.get("want_image", False))
    if want_image:
        try:
            img = generate_chart_placeholder(plan)
            if isinstance(img, bytes):
                import base64
                b64 = base64.b64encode(img).decode("ascii")
                return f"data:image/png;base64,{b64}"
            return str(img)
        except Exception as e:
            logger.warning("generate_chart_placeholder failed: %s", e)
            return "[VISUALIZATION_PLAN] " + sanitize_answer(json.dumps(plan))

    return "[VISUALIZATION_PLAN] " + sanitize_answer(json.dumps(plan))


# ==========================================================
# TEXT REASONING HANDLER
# ==========================================================
async def solve_text_reasoning_quiz(quiz_info: Dict[str, Any]) -> Any:
    text = quiz_info.get("question_text", "")
    url = quiz_info.get("url", "")

    prompt = f"""
URL: {url}

Question:
{text}

You MUST respond ONLY with the final answer, and nothing else.
If numeric, return a number. If yes/no, return true/false. If structured, return JSON.
"""
    raw = call_llm(prompt.strip(), system="Quiz solver (strict)")
    cleaned = sanitize_answer(raw)

    parsed = parse_possible_structured_answer(cleaned)
    if parsed is not None:
        return coerce_answer_type(parsed)

    return coerce_answer_type(cleaned)


# ==========================================================
# NETWORK HELPERS & SUBMISSION
# ==========================================================
def retry_post(url: str, json_payload: Dict[str, Any], retries: int = 2, backoff: float = 1.0, timeout: int = 20):
    last_exc = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(url, json=json_payload, timeout=timeout)
            resp.raise_for_status()
            return resp
        except Exception as e:
            last_exc = e
            logger.warning("POST attempt %d failed for %s: %s", attempt + 1, url, e)
            if attempt == retries:
                break
            time.sleep(backoff * (2 ** attempt))
    raise last_exc


def submit_answer(
    email: str,
    secret: str,
    quiz_url: str,
    answer: Any,
    submission_url: str,
    timeout: int = 20,
) -> Dict[str, Any]:
    """Send answer payload and return parsed JSON or structured error."""
    submission_url = urljoin(quiz_url, submission_url)

    payload = {
        "email": email,
        "secret": secret,
        "url": quiz_url,
        "answer": answer,
    }

    try:
        resp = retry_post(submission_url, json_payload=payload, retries=2, backoff=1.0, timeout=timeout)
    except Exception as e:
        logger.error("Failed to POST answer to %s: %s", submission_url, e)
        return {"correct": False, "url": None, "reason": f"network_error: {str(e)}"}

    try:
        return resp.json()
    except Exception:
        logger.error("Submission endpoint returned non-JSON response")
        return {"correct": False, "url": None, "reason": "invalid_json_response"}