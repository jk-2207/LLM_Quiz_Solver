from __future__ import annotations

import os
import asyncio
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from solver import solve_quiz_chain

app = FastAPI(title="LLM Quiz Solver")


# ==========================================================
# CONFIG
# ==========================================================
QUIZ_SECRET = os.getenv("QUIZ_SECRET", "jk_tds_2025_secret")


# ==========================================================
# ERROR HANDLERS
# ==========================================================
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # For invalid JSON or missing fields, return HTTP 400
    return JSONResponse(
        status_code=400,
        content={"detail": "Invalid JSON or missing required fields"},
    )


# ==========================================================
# REQUEST MODEL
# ==========================================================
class QuizRequest(BaseModel):
    email: str
    secret: str
    url: str


# ==========================================================
# HEALTH CHECK
# ==========================================================
@app.get("/")
async def root() -> Dict[str, Any]:
    return {"status": "ok", "message": "LLM Quiz Solver running"}


# ==========================================================
# QUIZ ENDPOINT
# ==========================================================
@app.post("/quiz")
async def quiz_endpoint(req: QuizRequest) -> Dict[str, Any]:
    # Secret validation
    if req.secret != QUIZ_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")

    # Solve quiz chain starting from the given URL
    result = await solve_quiz_chain(start_url=req.url, email=req.email, secret=req.secret)
    return result