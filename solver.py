from __future__ import annotations

import time
import re
import json
import logging
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, parse_qs

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
# LOGGING SETUP
# ==========================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("solver")

# ==========================================================
# SANITIZE LLM OUTPUT
# ==========================================================
import ast

def sanitize_answer(ans: Optional[str]) -> str:
    """Remove chain-of-thought blocks and common prefixes, return first meaningful line."""
    if not ans:
        return ""
    s = str(ans)

    # remove <think>...</think> and similar tags
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL | re.IGNORECASE)
    s = re.sub(r"<.*?reasoning.*?>.*?</.*?>", "", s, flags=re.DOTALL | re.IGNORECASE)

    # trim whitespace and split lines
    s = s.strip()
    lines = [l.strip() for l in s.splitlines() if l.strip()]
    if not lines:
        return ""

    first = lines[0]

    # common instruction prefixes to remove
    first = re.sub(r'(?i)^(final answer[:\-\s]*|answer[:\-\s]*|respond (only )?with[:\-\s]*)', '', first).strip()

    # Remove trailing ellipses and accidental trailing punctuation
    first = first.rstrip(". \t\n\r")

    # remove surrounding quotes
    first = first.strip().strip('"\'')

    return first

# ==========================================================
# MAIN SOLVER LOOP
# ==========================================================
def parse_possible_structured_answer(s: str):
    """
    Try to interpret 's' as JSON, else Python literal (single-quoted dict),
    else return None to indicate not-structured.
    Returns Python object on success, else None.
    """
    if not s:
        return None
    s = s.strip()
    # quick JSON attempt
    try:
        return json.loads(s)
    except Exception:
        pass

    # try to convert single-quoted dicts / Python literals safely using ast.literal_eval
    try:
        # ast.literal_eval will parse Python dict-like strings into Python objects
        obj = ast.literal_eval(s)
        # accept only simple types: dict/list/str/num/bool
        if isinstance(obj, (dict, list, str, int, float, bool)):
            return obj
    except Exception:
        pass

    # Try to extract a dict-like substring if the LLM padded text around it
    m = re.search(r"(\{.*\})", s, flags=re.DOTALL)
    if m:
        txt = m.group(1)
        try:
            return json.loads(txt)
        except Exception:
            try:
                return ast.literal_eval(txt)
            except Exception:
                pass

    return None

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

        logger.info(f"\n===== Solving quiz: {current_url} =====")
        html = fetch_html(current_url)
        quiz_info = parse_quiz_page(html, current_url)

        # Ensure email always present
        if not quiz_info.get("email"):
            quiz_info["email"] = email

        logger.info("Parsed quiz_info keys=" + str(list(quiz_info.keys())))

        try:
            answer = await compute_answer(quiz_info)
            logger.info(f"Computed answer: {answer}")
        except Exception as e:
            logger.exception("Error during compute_answer")
            history.append({"url": current_url, "error": str(e)})
            break

        resp = submit_answer(
            email=email,
            secret=secret,
            quiz_url=current_url,
            answer=answer,
            submission_url=quiz_info["submission_url"],
        )

        history.append({
            "url": current_url,
            "answer": answer,
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
        (s.get("type") == "module")
        or (s.get("src") or "").endswith(".js")
        for s in scripts
    )

    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    email = qs.get("email", [None])[0]

    submission_url = None

    # Case 1: <form action="...">
    form = soup.find("form")
    if form and form.get("action"):
        submission_url = form.get("action")

    # Case 2: scripts containing /submit
    if not submission_url:
        for tag in soup.find_all(["script", "a"], string=True):
            if tag.string and "/submit" in tag.string:
                m = re.search(r"https?://[^\"' ]+/submit", tag.string)
                if m:
                    submission_url = m.group(0)
                    break

    # Case 3: fallback
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

    # Strong JS detection
    if quiz_info.get("has_js_module"):
        return "js_scrape"

    # Explicit API cue
    if "api" in url or "fetch" in text or "endpoint" in text:
        return "api"

    # CSV / data / dataset
    if ".csv" in text or "dataset" in text or "data" in text:
        return "data_csv"

    # Visualization type cue
    if "chart" in text or "plot" in text or "visualize" in text:
        return "visualization"

    # LLM fallback classification
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
    logger.info(f"Detected type: {quiz_type}")

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
    except Exception:
        # fallback
        html = fetch_html(url)
        soup = BeautifulSoup(html, "html.parser")

    q_div = soup.find("div", {"id": "question"})
    text = q_div.get_text(" ", strip=True) if q_div else soup.get_text(" ", strip=True)

    if not text.strip():
        raise ValueError("JS-rendered page has no visible text")

    # Detect secret codes
    m = re.search(r"(secret|code)\s*[:\- ]*\s*([0-9]{2,})", text, flags=re.IGNORECASE)
    if m:
        return sanitize_answer(m.group(2))

    # Detect if JS generated JSON in page
    mjson = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if mjson:
        try:
            jsdata = json.loads(mjson.group(0))
            if isinstance(jsdata, dict) and "secret" in jsdata:
                return sanitize_answer(str(jsdata["secret"]))
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
# API HANDLER (Unified Execution)
# ==========================================================
async def solve_api_quiz(quiz_info: Dict[str, Any]) -> Any:
    text = quiz_info["question_text"]

    prompt = f"""
    You are an expert API analyzer.
    Read the question and produce a JSON instruction plan with these fields:

    {{
      "endpoint": "API URL",
      "method": "GET or POST",
      "params": {{ ... }},
      "extract": "description of value to extract",
      "answer_type": "number | string | list | json"
    }}

    QUESTION:
    {text}

    Respond ONLY with valid JSON.
    """
    raw = call_llm(prompt.strip(), system="API Planner")
    js = None
    try:
        js = json.loads(sanitize_answer(raw))
    except Exception:
        raise ValueError(f"LLM returned invalid API JSON: {raw}")

    endpoint = js.get("endpoint")
    method = js.get("method", "GET").upper()
    params = js.get("params", {})

    if not endpoint:
        raise ValueError("API plan missing endpoint")

    data = fetch_api_json(endpoint, method=method, params=params)

    # extract using simple key lookup
    extract_key = js.get("extract")
    if extract_key and extract_key in data:
        return sanitize_answer(str(data[extract_key]))

    return sanitize_answer(str(data))

# ==========================================================
# NETWORK / SUBMIT HELPERS
# ==========================================================
def retry_post(url: str, json_payload: Dict[str, Any], retries: int = 2, backoff: float = 1.0, timeout: int = 20):
    """
    Simple retry wrapper for requests.post. Raises last exception on failure.
    """
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
    """
    Send the answer payload to the submission endpoint and return parsed JSON.
    Returns a structured dict with keys: correct (bool), url (next URL or None), reason (str).
    On network failure, returns a structured error dict instead of raising to the main loop.
    """
    # import urljoin locally in case upper parts didn't import it
    from urllib.parse import urljoin

    # Ensure submission_url is absolute relative to quiz_url
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

    # Parse JSON response
    try:
        return resp.json()
    except Exception:
        logger.error("Submission endpoint returned non-JSON response")
        return {"correct": False, "url": None, "reason": "invalid_json_response"}


# ==========================================================
# CSV / DATA QUIZ HANDLER
# ==========================================================
async def solve_data_quiz(quiz_info: Dict[str, Any]) -> Any:
    """
    CSV solver with heuristics + LLM JSON plan fallback.
    Returns sanitized final answer (string or JSON-string).
    """
    text = (quiz_info.get("question_text") or "").strip()
    if not text:
        raise ValueError("Empty question text for data quiz.")

    # 1) find CSV URL
    match = re.search(r"(https?://\S+?\.csv)(?:\s|$)", text, re.IGNORECASE)
    if not match:
        # maybe the page contains a CSV link not in text; try fetching the page
        page_html = fetch_html(quiz_info.get("url", ""))
        m2 = None
        if page_html:
            m2 = re.search(r"(https?://\S+?\.csv)(?:\s|<|\")", page_html, re.IGNORECASE)
        if m2:
            match = m2

    if not match:
        raise ValueError("No CSV URL found in question.")

    csv_url = match.group(1)

    # 2) load CSV
    try:
        df = load_csv_from_url(csv_url)
    except Exception as e:
        raise ValueError(f"Failed to load CSV: {e}")

    if df is None or df.empty:
        raise ValueError("Loaded CSV is empty or unreadable.")

    # helper to pick a numeric column if none provided
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

    # group-by pattern: "group by <col> sum <col2>" or "sum <col2> by <col>"
    gb_match = re.search(r"(?:group by|groupby)\s+([A-Za-z0-9_ \-]+).*?(?:sum|total|aggregate)\s+([A-Za-z0-9_ \-]+)", lower)
    if gb_match:
        col_group = gb_match.group(1).strip()
        col_value = gb_match.group(2).strip()
        # map to actual column names (case-insensitive)
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

    # sum / total of a column
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
            return sanitize_answer(str(val))
        except Exception as e:
            raise ValueError(f"sum failed: {e}")

    # mean / average
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
            return sanitize_answer(str(val))
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
            return sanitize_answer(str(val))
        except Exception as e:
            raise ValueError(f"median failed: {e}")

    # count rows or count distinct
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
                return sanitize_answer(str(val))
            except Exception as e:
                raise ValueError(f"distinct count failed: {e}")
        # otherwise total rows
        return sanitize_answer(str(len(df)))

    # max / min
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
            return sanitize_answer(str(val))
        except Exception as e:
            raise ValueError(f"max failed: {e}")

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
            return sanitize_answer(str(val))
        except Exception as e:
            raise ValueError(f"min failed: {e}")

    # Fallback: ask LLM for machine-readable plan (JSON)
    llm_prompt = f"""
You are a data assistant. CSV: {csv_url}
Question: {text}

Return a tiny JSON object (ONLY JSON) describing what to compute, example:
{{"operation":"sum", "column":"Sales", "filter": "Region=='APAC'"}}
Allowed operations: sum, mean, median, count, distinct_count, max, min, groupby_sum
For groupby_sum use keys: "groupby":"ColumnA", "agg":"ColumnB"
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
            return sanitize_answer(str(df[col].sum()))

        if op == "mean":
            col = plan.get("column") or _pick_numeric_column()
            return sanitize_answer(str(df[col].mean()))

        if op == "median":
            col = plan.get("column") or _pick_numeric_column()
            return sanitize_answer(str(df[col].median()))

        if op == "count":
            if plan.get("column"):
                col = plan["column"]
                return sanitize_answer(str(int(df[col].count())))
            return sanitize_answer(str(len(df)))

        if op == "distinct_count":
            col = plan.get("column")
            return sanitize_answer(str(int(df[col].nunique())))

        if op == "max":
            col = plan.get("column") or _pick_numeric_column()
            return sanitize_answer(str(df[col].max()))

        if op == "min":
            col = plan.get("column") or _pick_numeric_column()
            return sanitize_answer(str(df[col].min()))

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
If the plan requests an image, set "want_image": true. Respond with valid JSON only.
"""
    raw = call_llm(prompt.strip(), system="Visualization assistant (json)")
    cleaned = sanitize_answer(raw)

    try:
        plan = json.loads(cleaned)
    except Exception:
        return "[VISUALIZATION_PLAN] " + cleaned

    # If want_image and helper exists, attempt to create an image
    want_image = bool(plan.get("want_image", False))
    if want_image:
        try:
            # generate_chart_placeholder(plan) should return bytes or a base64 string/uri
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
# TEXT / REASONING QUIZ
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

    low = cleaned.lower()
    if low in {"true", "false", "yes", "no"}:
        return True if low in {"true", "yes"} else False

    # integer
    if re.fullmatch(r"[-+]?\d+", cleaned):
        try:
            return int(cleaned)
        except Exception:
            pass

    # float
    if re.fullmatch(r"[-+]?\d*\.\d+(e[-+]?\d+)?", cleaned, flags=re.I) or re.fullmatch(r"[-+]?\d+(\.\d*)?[eE][-+]?\d+", cleaned):
        try:
            return float(cleaned)
        except Exception:
            pass

    # JSON
    try:
        parsed = json.loads(cleaned)
        return parsed
    except Exception:
        pass

    return cleaned