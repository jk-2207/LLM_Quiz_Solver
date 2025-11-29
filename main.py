# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

from solver import solve_quiz_chain

app = FastAPI(title="TDS Quiz Solver – Orchestrator")

QUIZ_SECRET = os.getenv("QUIZ_SECRET", "jk_tds_2025_secret")
QUIZ_EMAIL = os.getenv("QUIZ_EMAIL", "24f3000312@ds.study.iitm.ac.in")


class QuizRequest(BaseModel):
    email: str
    secret: str
    url: str


@app.post("/quiz")
async def quiz_endpoint(data: QuizRequest):
    # Secret validation
    if data.secret != QUIZ_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")

    if not data.email or not data.url:
        raise HTTPException(status_code=400, detail="Invalid payload")

    try:
        result = await solve_quiz_chain(
            start_url=data.url,
            email=data.email,
            secret=data.secret,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return result