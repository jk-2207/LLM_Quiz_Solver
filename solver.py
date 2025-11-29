# solver.py
from __future__ import annotations

import time
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, parse_qs, urljoin

import requests
from bs4 import BeautifulSoup  # type: ignore

from tools import (
    fetch_html,
    render_with_js,
    fetch_api_json,
    load_csv_from_url,
    basic_groupby_sum,
    generate_chart_placeholder,
)
from llm_client import call_llm


# ==========================================================
# HELPERS
# ==========================================================

def _clean_label(raw: str) -> str:
    """Return a cleaned classifier label (first token, stripped punctuation)."""
    if not raw:
        return ""
    token = raw.strip().split()[0].lower().strip('.,;"\'()[]')
    return token


def sanitize_answer(ans: Optional[str]) -> str:
    """
    Normalize LLM outputs to a concise final answer string.
    - Remove <think>...</think> blocks
    - Take the first non-empty line.
    - Strip common prefixes like "answer is".
    - Remove surrounding quotes.
    """
    if not ans:
        return ""
    s = str(ans)
    # remove any <think>...</think> or similar blocks
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL | re.IGNORECASE)
    s = s.strip()
    lines = [l.strip() for l in s.splitlines() if l.strip()]
    if not lines:
        return ""
    first = lines[0]
    # Remove common prefixes
    first = re.sub(r'(?i)^(answer[:\s-]*|the answer (is|:)\s*)', '', first).strip()
    first = first.strip().strip('"\'' )
    return first


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
            if attempt == retries:
                raise
            time.sleep(backoff * (2 ** attempt))
    raise last_exc


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

        html = fetch_html(current_url)
        if not html:
            history.append({"url": current_url, "error": "failed_fetch_html"})
            break

        quiz_info = parse_quiz_page(html, current_url)

        # Ensure email is always present
        if not quiz_info.get("email"):
            quiz_info["email"] = email

        try:
            answer = await compute_answer(quiz_info)
        except Exception as e:
            history.append({"url": current_url, "error": str(e)})
            break

        # sanitize answer before submission
        safe_answer = sanitize_answer(answer)

        resp = submit_answer(
            email=email,
            secret=secret,
            quiz_url=current_url,
            answer=safe_answer,
            submission_url=quiz_info["submission_url"],
        )

        history.append({
            "url": current_url,
            "answer": safe_answer,
            "correct": resp.get("correct", False),
            "next_url": resp.get("url"),
            "reason": resp.get("reason", "")
        })

        # Only increment solved when the judge says correct==true
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
    """
    Parse the quiz page to extract:
    - question_text
    - has_js_module flag
    - submission_url (absolute if possible)
    - email (if present in URL)
    """
    soup = BeautifulSoup(html, "html.parser")

    # Extract question
    question_el = (
        soup.find("div", {"id": "question"})
        or soup.find("div", {"class": "question"})
        or soup.find("h1")
    )
    question_text = question_el.get_text(strip=True) if question_el else ""

    # Detect JS-heavy page
    scripts = soup.find_all("script")
    has_js_module = any(
        (s.get("type") == "module") or (s.get("src") or "").endswith(".js")
        for s in scripts
    )

    # Extract email from URL if present
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    email = qs.get("email", [None])[0]

    # ----------------- Detect submission URL -----------------
    submission_url = None

    # Case 1: <form action="...">
    form = soup.find("form")
    if form and form.get("action"):
        submission_url = urljoin(url, form.get("action"))

    # Case 2: check <a href="..."> links and button formaction
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

    # Case 3: look inside script / text nodes for an absolute /submit URL
    if not submission_url:
        for tag in soup.find_all(["script", "div", "span"]):
            txt = ""
            if tag.string:
                txt = tag.string
            else:
                txt = tag.get_text(" ", strip=True) or ""
            if "/submit" in txt:
                match = re.search(r"https?://[^\"' ]+/submit", txt)
                if match:
                    submission_url = match.group(0)
                    break

    # Case 4: fallback to same domain + /submit
    if not submission_url:
        parsed_base = urlparse(url)
        submission_url = f"{parsed_base.scheme}://{parsed_base.netloc}/submit"

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
    has_js_module = quiz_info.get("has_js_module", False)

    # Prefer JS-scrape for JS-heavy pages
    if has_js_module:
        return "js_scrape"

    if "api" in url or "api" in text:
        return "api"

    if "csv" in text or "dataset" in text or ".csv" in text:
        return "data_csv"

    if "chart" in text or "plot" in text or "visualize" in text:
        return "visualization"

    # LLM fallback classification
    prompt = f"""
    Classify this quiz question:
    Text: {quiz_info.get("question_text")}
    URL: {quiz_info.get("url")}

    Choose one of: js_scrape, api, data_csv, visualization, text_reasoning
    """
    raw = call_llm(prompt.strip(), system="Classifier")
    label = _clean_label(raw)
    if label not in {"js_scrape", "api", "data_csv", "visualization", "text_reasoning"}:
        return "text_reasoning"
    return label


# ==========================================================
# DISPATCHER
# ==========================================================

async def compute_answer(quiz_info: Dict[str, Any]) -> Any:
    quiz_type = detect_quiz_type(quiz_info)

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
# HANDLERS
# ==========================================================

async def solve_js_scrape(quiz_info: Dict[str, Any]) -> Any:
    url = quiz_info["url"]

    try:
        # Try full JS rendering
        rendered_html = await render_with_js(url)
        soup = BeautifulSoup(rendered_html, "html.parser")
    except Exception:
        # Fallback: no JS, just plain HTML
        html = fetch_html(url)
        soup = BeautifulSoup(html, "html.parser")

    q_div = soup.find("div", {"id": "question"})
    text = q_div.get_text(" ", strip=True) if q_div else soup.get_text(" ", strip=True)

    if not text:
        raise ValueError("JS-rendered page has no visible text")

    # Try obvious "secret/code" pattern
    match = re.search(r"(secret|code)\s*[:\s]*([0-9]+)", text, re.IGNORECASE)
    if match:
        return match.group(2)

    # Fallback: let the LLM extract the answer from the rendered text
    prompt = f"""
    Extract the final answer from this content:

    {text}

    Respond ONLY with the answer value.
    """
    answer = call_llm(prompt.strip(), system="Answer extractor")
    return sanitize_answer(answer)


async def solve_api_quiz(quiz_info: Dict[str, Any]) -> Any:
    """
    Attempt to solve API-style quizzes.
    Strategy:
    1. Look for explicit API URL in question_text (regex).
    2. If found, call it (fetch_api_json) and return sensible value (string/number/json).
    3. If not found, inspect the rendered page (try render_with_js) for example endpoint/headers.
    4. If still nothing, ask the LLM but instruct it to return ONLY the final answer (JSON/plain).
    """
    text = quiz_info.get("question_text", "") or ""
    url = quiz_info.get("url")

    # 1) quick regex for obvious API URLs in the question text
    m = re.search(r"(https?://[^\s'\"<>{}]+/[^)\s'\"<>{}]+)", text)
    if m:
        candidate = m.group(1)
        # If it looks like an API endpoint (has 'api' or ends with /json etc.), try it
        if "api" in candidate.lower() or candidate.lower().endswith((".json", "/data")):
            try:
                data = fetch_api_json(candidate)
                # If it's a simple JSON with result key, prefer it; otherwise return full JSON
                if isinstance(data, dict):
                    # prefer 'answer' or 'result' keys if present
                    for k in ("answer", "result", "data", "value"):
                        if k in data:
                            return sanitize_answer(str(data[k]))
                return sanitize_answer(str(data))
            except Exception as e:
                # fallthrough to LLM planning if direct call failed
                pass

    # 2) try rendering page (JS) to find endpoint examples / headers
    try:
        rendered = await render_with_js(url)
        soup = BeautifulSoup(rendered, "html.parser")
        page_text = soup.get_text(" ", strip=True)
        # look for obvious example endpoints in page text
        m2 = re.search(r"https?://[^\s'\"<>{}]+/submit[^\s'\"<>]*", page_text)
        if m2:
            candidate = m2.group(0)
            try:
                data = fetch_api_json(candidate)
                if isinstance(data, dict):
                    for k in ("answer","result","data","value"):
                        if k in data:
                            return sanitize_answer(str(data[k]))
                return sanitize_answer(str(data))
            except Exception:
                pass
    except Exception:
        # rendering failed — no big deal, fallback to LLM
        pass

    # 3) fallback: ask the LLM but force a strict output format
    llm_prompt = f"""
You are an assistant that must *only* answer with the final result (no explanation).
The user asked: {text}
If you need to propose an API endpoint, return a compact JSON object describing:
{{"endpoint":"<url>", "method":"GET/POST", "headers":{{...}} , "answer_type":"number|string|json", "example_answer": "<the final answer>" }}
Respond with valid JSON only and nothing else.
"""
    raw = call_llm(llm_prompt.strip(), system="API planner (strict)")
    # sanitize then try to parse JSON
    cleaned = sanitize_answer(raw)
    # try to extract JSON
    try:
        import json
        parsed = json.loads(cleaned)
        # if we have example_answer, return it
        if isinstance(parsed, dict) and "example_answer" in parsed:
            return sanitize_answer(str(parsed["example_answer"]))
        # otherwise return the whole JSON as string
        return sanitize_answer(str(parsed))
    except Exception:
        # if parsing failed, just return the sanitized string (best-effort)
        return cleaned

async def solve_data_quiz(quiz_info: Dict[str, Any]) -> Any:
    """
    Improved CSV solver.

    Strategy:
    - Look for CSV URL in question text.
    - Load CSV using your load_csv_from_url helper (returns a pandas.DataFrame).
    - Try simple heuristics to detect operations (sum, mean, median, count, max, min, groupby).
    - If heuristics fail, ask the LLM for a tiny JSON plan and execute it.
    - Return the computed scalar or a compact JSON result when appropriate.
    """
    import json
    from pandas.api.types import is_numeric_dtype

    text = (quiz_info.get("question_text") or "").strip()
    if not text:
        raise ValueError("Empty question text for data quiz.")

    # 1) find CSV URL
    match = re.search(r"(https?://\S+?\.csv)(?:\s|$)", text, re.IGNORECASE)
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

    # helper to pick a likely numeric column if none provided
    def _pick_numeric_column():
        for c in df.columns:
            if is_numeric_dtype(df[c]):
                return c
        return None

    # 3) simple heuristics
    lower = text.lower()

    # group-by pattern: "group by <col> sum <col2>" or "sum <col2> by <col>"
    gb_match = re.search(r"(?:group by|groupby)\s+([A-Za-z0-9_ -]+).*?(?:sum|total|aggregate)\s+([A-Za-z0-9_ -]+)", lower)
    if gb_match:
        col_group = gb_match.group(1).strip()
        col_value = gb_match.group(2).strip()
        # try to map to actual column names (case-insensitive)
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
    sum_match = re.search(r"(?:sum|total|add|sum of)\s+([A-Za-z0-9_ -]+)", lower)
    if sum_match:
        col_name = sum_match.group(1).strip()
        # map to actual column
        col = None
        for c in df.columns:
            if c.lower() == col_name.lower() or col_name.lower() in c.lower():
                col = c
                break
        if not col:
            # fallback: pick a numeric column
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
        # try to detect column
        mean_match = re.search(r"(?:mean|average|avg)\s+of\s+([A-Za-z0-9_ -]+)", lower)
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
        med_match = re.search(r"median\s+of\s+([A-Za-z0-9_ -]+)", lower)
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
        # distinct pattern: "count distinct <col>"
        distinct_match = re.search(r"(?:count distinct|distinct count)\s+([A-Za-z0-9_ -]+)", lower)
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
        # detect column
        m = re.search(r"(?:max|maximum)\s+of\s+([A-Za-z0-9_ -]+)", lower)
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
        m = re.search(r"(?:min|minimum)\s+of\s+([A-Za-z0-9_ -]+)", lower)
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

    # 4) If heuristics failed, ask LLM for a tiny JSON plan
    llm_prompt = f"""
You are a helpful data assistant. Given the CSV available at: {csv_url}
Question: {text}

Return a tiny JSON object (and ONLY the JSON) describing what to compute. Example:
{{"operation":"sum", "column":"Sales", "filter": "Region=='APAC'"}}
Allowed operations: sum, mean, median, count, distinct_count, max, min, groupby_sum
If you propose groupby_sum use keys: "groupby":"ColumnA", "agg":"ColumnB"

Respond with valid JSON only.
"""
    raw = call_llm(llm_prompt.strip(), system="Data planner (json)")
    cleaned = sanitize_answer(raw)

    try:
        plan = json.loads(cleaned)
    except Exception:
        # cannot parse LLM output
        raise ValueError("LLM returned unparseable plan for CSV. Plan text: " + cleaned)

    # Execute the plan
    op = plan.get("operation")
    try:
        if op == "sum":
            col = plan.get("column") or _pick_numeric_column()
            if col not in df.columns:
                # try fuzzy match
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
            import json as _json
            return sanitize_answer(_json.dumps(res))

        # Unknown op
        raise ValueError(f"Unknown operation requested by plan: {op}")
    except Exception as e:
        raise ValueError(f"Failed to execute plan: {e}")

async def solve_visualization_quiz(quiz_info: Dict[str, Any]) -> Any:
    text = quiz_info.get("question_text", "")

    prompt = f"""
    Question:
    {text}

    Suggest visualization type and axes.
    """
    chart_plan = call_llm(prompt.strip(), system="Visualization assistant")

    return f"[VISUALIZATION_PLAN] {chart_plan}"


async def solve_text_reasoning_quiz(quiz_info: Dict[str, Any]) -> Any:
    text = quiz_info.get("question_text", "")
    url = quiz_info.get("url", "")

    prompt = f"""
    URL: {url}

    Question:
    {text}

    Respond ONLY with the final answer.
    """
    answer = call_llm(prompt.strip(), system="Quiz solver")

    return sanitize_answer(answer)


# ==========================================================
# SUBMISSION FUNCTION
# ==========================================================

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
    On network failure, returns a structured error dict instead of raising.
    """
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
        return {"correct": False, "url": None, "reason": f"network_error: {str(e)}"}

    # Parse JSON response
    try:
        return resp.json()
    except ValueError:
        return {"correct": False, "url": None, "reason": "invalid_json_response"}