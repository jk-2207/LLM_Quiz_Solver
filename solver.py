# solver.py
from __future__ import annotations

import time
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, parse_qs

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
        quiz_info = parse_quiz_page(html, current_url)

        # Ensure email is always present
        if not quiz_info.get("email"):
            quiz_info["email"] = email

        try:
            answer = await compute_answer(quiz_info)
        except Exception as e:
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

        # ✅ Only count as solved if correct
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
        submission_url = form.get("action")

    # Case 2: JavaScript or text containing /submit
    if not submission_url:
        for tag in soup.find_all(["a", "script"], string=True):
            if tag.string and "/submit" in tag.string:
                match = re.search(r"https?://[^\"' ]+/submit", tag.string)
                if match:
                    submission_url = match.group(0)
                    break

    # Case 3: Fallback to same domain + /submit
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

    # ✅ Always treat JS-heavy pages as js_scrape
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
    label = raw.strip().split()[0].lower()

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
        rendered_html = await render_with_js(url)
    except NotImplementedError:
        # Fallback: just use plain HTML if Playwright can’t run
        html = fetch_html(url)
        soup = BeautifulSoup(html, "html.parser")
    else:
        soup = BeautifulSoup(rendered_html, "html.parser")

    q_div = soup.find("div", {"id": "question"})
    text = q_div.get_text(" ", strip=True) if q_div else soup.get_text(" ", strip=True)

    if not text:
        raise ValueError("JS-rendered page has no visible text")

    match = re.search(r"(secret|code)\s*[:\s]*([0-9]+)", text, re.IGNORECASE)
    if match:
        return match.group(2)

    prompt = f"""
    Extract the final answer from this content:

    {text}

    Respond ONLY with the answer value.
    """
    answer = call_llm(prompt.strip(), system="Answer extractor")
    return answer.strip().splitlines()[0].strip()


async def solve_api_quiz(quiz_info: Dict[str, Any]) -> Any:
    text = quiz_info.get("question_text", "")

    prompt = f"""
    You are planning an API-based solution.

    Question:
    {text}

    Describe expected API endpoint + final answer type.
    """
    plan = call_llm(prompt.strip(), system="API planner")

    raise ValueError(f"API solver not implemented yet. Plan suggestion: {plan}")


async def solve_data_quiz(quiz_info: Dict[str, Any]) -> Any:
    text = quiz_info.get("question_text", "")
    match = re.search(r"(https?://\S+\.csv)", text)

    if not match:
        raise ValueError("No CSV URL found in question.")

    csv_url = match.group(1)
    df = load_csv_from_url(csv_url)

    prompt = f"""
    Dataset columns: {list(df.columns)}

    Question: {text}

    Suggest pandas steps to compute answer.
    """
    plan = call_llm(prompt.strip(), system="Data planner")

    raise ValueError(f"CSV solver not implemented yet. Plan suggestion: {plan}")


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

    return answer.strip().splitlines()[0].strip()


# ==========================================================
# SUBMISSION FUNCTION
# ==========================================================

def submit_answer(
    email: str,
    secret: str,
    quiz_url: str,
    answer: Any,
    submission_url: str
) -> Dict[str, Any]:

    payload = {
        "email": email,
        "secret": secret,
        "url": quiz_url,
        "answer": answer,
    }

    response = requests.post(submission_url, json=payload, timeout=20)
    response.raise_for_status()

    return response.json()