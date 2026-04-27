from __future__ import annotations

import csv
import io
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from fastapi import Body, FastAPI
from fastapi.responses import FileResponse, JSONResponse, Response

from .llm_config import LLMClient, create_env_template

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"

app = FastAPI(title="VectorQuiz API")
_client: LLMClient | None = None


def get_client() -> LLMClient:
    """Lazily initialize the LLM client so startup works without keys."""
    global _client
    if _client is None:
        _client = LLMClient()
    return _client


def json_error(message: str, status: int = 400) -> JSONResponse:
    return JSONResponse(status_code=status, content={"ok": False, "error": message})


def _validate_quiz_payload(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    questions = payload.get("questions")
    if not isinstance(questions, list) or not questions:
        raise ValueError("questions must be a non-empty array.")
    return questions


def _compute_result(
    questions: List[Dict[str, Any]], answers: Dict[str, str]
) -> Dict[str, Any]:
    breakdown = []
    correct = 0

    for idx, question in enumerate(questions, 1):
        q_key = f"q_{idx}"
        user_answer = str(answers.get(q_key, "")).strip().upper()
        correct_answer = str(question.get("correct_answer", "")).strip().upper()
        is_correct = user_answer == correct_answer
        if is_correct:
            correct += 1

        breakdown.append(
            {
                "question_no": idx,
                "your_answer": user_answer,
                "correct_answer": correct_answer,
                "is_correct": is_correct,
            }
        )

    total = len(questions)
    percent = round((correct / total) * 100) if total else 0
    return {
        "correct": correct,
        "total": total,
        "percent": percent,
        "breakdown": breakdown,
    }


def _export_csv(questions: List[Dict[str, Any]]) -> str:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["question", "correct_answer", "explanation", "options"])
    for q in questions:
        writer.writerow(
            [
                q.get("question", ""),
                q.get("correct_answer", ""),
                q.get("explanation", ""),
                json.dumps(q.get("options", {}), ensure_ascii=False),
            ]
        )
    return output.getvalue()


def _export_txt(topic: str, questions: List[Dict[str, Any]]) -> str:
    lines: List[str] = [f"QUIZ: {topic}", ""]
    for idx, q in enumerate(questions, 1):
        lines.append(f"Q{idx}: {q.get('question', '')}")
        options = q.get("options", {})
        if isinstance(options, dict):
            for key, value in options.items():
                lines.append(f"  {key}. {value}")
        lines.append(f"  Answer: {q.get('correct_answer', '')}")
        lines.append(f"  Explanation: {q.get('explanation', '')}")
        lines.append("")
    return "\n".join(lines)


@app.get("/")
def home() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/health")
def health() -> Dict[str, bool]:
    return {"ok": True}


@app.get("/api/models")
def models() -> Response:
    try:
        client = get_client()
    except ValueError as exc:
        return json_error(str(exc), 400)

    available = [
        {
            "id": model.id,
            "name": model.name,
            "description": model.description,
            "provider": model.provider,
        }
        for model in client.available_models
    ]
    return JSONResponse(
        content={
            "ok": True,
            "models": available,
            "provider_status": client.provider_status(),
            "current_model": client.current_model.id,
        }
    )


@app.post("/api/create-env-template")
def create_env() -> Dict[str, Any]:
    create_env_template()
    return {"ok": True, "message": ".env template generated."}


@app.post("/api/quiz")
def generate_quiz(payload: Dict[str, Any] | None = Body(default=None)) -> Response:
    data = payload or {}

    topic = str(data.get("topic", "")).strip()
    model_id = str(data.get("model", "")).strip()

    try:
        n_questions = int(data.get("n_questions", 10))
        choices = int(data.get("choices", 4))
    except (TypeError, ValueError):
        return json_error("n_questions and choices must be integers.")

    difficulty = str(data.get("difficulty", "medium")).strip().lower()

    if not topic:
        return json_error("Topic is required.")
    if n_questions < 1 or n_questions > 50:
        return json_error("n_questions must be between 1 and 50.")
    if choices < 3 or choices > 5:
        return json_error("choices must be between 3 and 5.")
    if difficulty not in {"easy", "medium", "hard"}:
        return json_error("difficulty must be one of: easy, medium, hard.")

    try:
        client = get_client()
        if model_id:
            client.set_model(model_id)
        mcqs: List[Dict[str, Any]] = client.generate_mcqs(
            topic=topic,
            n=n_questions,
            difficulty=difficulty,
            choices=choices,
        )
    except ValueError as exc:
        return json_error(str(exc), 400)
    except RuntimeError as exc:
        return json_error(str(exc), 502)
    except Exception as exc:  # noqa: BLE001
        return json_error(f"Unexpected server error: {exc}", 500)

    return JSONResponse(
        content={
            "ok": True,
            "topic": topic,
            "difficulty": difficulty,
            "model": client.current_model.id,
            "questions": mcqs,
        }
    )


@app.post("/api/quiz/submit")
def submit_quiz(payload: Dict[str, Any] | None = Body(default=None)) -> Response:
    data = payload or {}

    try:
        questions = _validate_quiz_payload(data)
    except ValueError as exc:
        return json_error(str(exc), 400)

    answers = data.get("answers", {})
    if not isinstance(answers, dict):
        return json_error("answers must be an object keyed by question id.", 400)

    result = _compute_result(questions, answers)
    return JSONResponse(content={"ok": True, "result": result})


@app.post("/api/quiz/export/{fmt}")
def export_quiz(fmt: str, payload: Dict[str, Any] | None = Body(default=None)) -> Response:
    data = payload or {}

    try:
        questions = _validate_quiz_payload(data)
    except ValueError as exc:
        return json_error(str(exc), 400)

    topic = str(data.get("topic", "quiz")).strip() or "quiz"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if fmt == "json":
        body = json.dumps(questions, indent=2, ensure_ascii=False)
        mimetype = "application/json"
        filename = f"{topic}_{timestamp}.json"
    elif fmt == "csv":
        body = _export_csv(questions)
        mimetype = "text/csv"
        filename = f"{topic}_{timestamp}.csv"
    elif fmt == "txt":
        body = _export_txt(topic, questions)
        mimetype = "text/plain"
        filename = f"{topic}_{timestamp}.txt"
    else:
        return json_error("Unsupported format. Use json, csv, or txt.", 400)

    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return Response(content=body, media_type=mimetype, headers=headers)


@app.get("/{path:path}")
def static_proxy(path: str) -> FileResponse:
    candidate = FRONTEND_DIR / path
    if candidate.exists() and candidate.is_file():
        return FileResponse(candidate)
    return FileResponse(FRONTEND_DIR / "index.html")
