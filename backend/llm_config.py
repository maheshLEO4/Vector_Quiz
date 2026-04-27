# llm_config.py
import os
import json
import requests
from typing import List, Dict, Optional
from dataclasses import dataclass, field

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass


# ── Model Registry ────────────────────────────────────────────────────────────

@dataclass
class Model:
    id: str
    name: str
    description: str
    provider: str          # 'gemini' | 'groq'
    max_tokens: int = 4096
    temperature: float = 0.7


_MODELS: List[Model] = [
    # Groq (Default)
    Model("llama-3.1-8b-instant",   "Llama 3.1 8B Instant",    "Groq · fastest",              "groq"),

    # Gemini
    Model("gemini-2.0-flash",       "Gemini 2.0 Flash",       "Fast & balanced",            "gemini"),
    Model("gemini-2.5-flash",       "Gemini 2.5 Flash",       "Smart & efficient",          "gemini"),
    Model("gemini-2.0-flash-lite",  "Gemini 2.0 Flash Lite",  "Ultra-fast, lightweight",    "gemini"),

    # Gemini 3.1
    Model("gemini-3.1-flash",       "Gemini 3.1 Flash",       "Fast Gemini 3 model",         "gemini"),
    Model("gemini-3.1-pro",         "Gemini 3.1 Pro",         "Advanced reasoning model",    "gemini"),

    # Other Groq
    Model("llama-3.3-70b-versatile","Llama 3.3 70B",          "Groq · powerful",             "groq"),
    Model("mixtral-8x7b-32768",     "Mixtral 8×7B",            "Groq · strong reasoning",     "groq"),
    Model("gemma2-9b-it",           "Gemma 2 9B",              "Groq · open model",            "groq"),
]
MODEL_MAP: Dict[str, Model] = {m.id: m for m in _MODELS}


# ── Prompt ────────────────────────────────────────────────────────────────────

def _build_prompt(topic: str, n: int, difficulty: str, choices: int) -> str:
    letters = "ABCDE"[:choices]
    options_block = "\n".join(f'      "{l}": "..."' for l in letters)
    return f"""Generate exactly {n} multiple-choice questions about: \"{topic}\"

Rules:
- Difficulty: {difficulty}
- Exactly {choices} options per question ({", ".join(letters)})
- One correct answer per question
- Factually accurate, educationally sound
- Test understanding, not just recall
- Plausible but wrong distractors

Return ONLY a valid JSON array — no markdown, no commentary:
[
  {{
    \"question\": \"...\",
    \"options\": {{
{options_block}
    }},
    \"correct_answer\": \"{letters[0]}\",
    \"explanation\": \"...\"
  }}
]"""

_SYSTEM = "You are an expert quiz creator. Output ONLY valid JSON arrays. No markdown, no extra text."


# ── Response Parser ───────────────────────────────────────────────────────────

def _parse(text: str) -> List[Dict]:
    """Extract a JSON list from model output."""
    t = text.strip()
    # Strip markdown fences
    for fence in ("```json", "```"):
        if t.startswith(fence):
            t = t[len(fence):]
    t = t.strip().rstrip("`").strip()

    # Slice out the JSON array
    s, e = t.find("["), t.rfind("]")
    if s != -1 and e > s:
        try:
            result = json.loads(t[s:e + 1])
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass
    # Last resort
    try:
        result = json.loads(t)
        return result if isinstance(result, list) else []
    except json.JSONDecodeError:
        return []


def _validate(mcq: Dict, choices: int) -> Optional[Dict]:
    """Validate and normalise one MCQ."""
    if not isinstance(mcq, dict):
        return None
    q = str(mcq.get("question", "")).strip()
    if len(q) < 5:
        return None
    opts = mcq.get("options")
    if not isinstance(opts, dict):
        return None
    clean_opts = {
        k.upper(): str(v).strip()
        for k, v in opts.items()
        if k.upper() in "ABCDE" and str(v).strip()
    }
    if len(clean_opts) < 2:
        return None
    correct = str(mcq.get("correct_answer", "")).strip().upper()
    if correct not in clean_opts:
        correct = list(clean_opts.keys())[0]
    return {
        "question": q,
        "options": clean_opts,
        "correct_answer": correct,
        "explanation": str(mcq.get("explanation", "No explanation provided.")).strip(),
    }


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _http_post(url: str, payload: Dict, headers: Dict = None, timeout: int = 45) -> Dict:
    """POST with unified error handling; returns parsed JSON."""
    try:
        r = requests.post(url, json=payload, headers=headers or {}, timeout=timeout)
    except requests.exceptions.Timeout:
        raise RuntimeError("Request timed out — please try again.")
    except requests.exceptions.ConnectionError:
        raise RuntimeError("Network error — check your internet connection.")

    if r.status_code == 429:
        raise RuntimeError("Rate limit hit — wait a moment then retry.")
    if r.status_code in (401, 403):
        raise RuntimeError("API key rejected — check your key in the .env file.")
    if not r.ok:
        try:
            msg = r.json().get("error", {}).get("message", r.text[:200])
        except Exception:
            msg = r.text[:200]
        raise RuntimeError(f"API error {r.status_code}: {msg}")

    return r.json()


# ── Provider Clients ──────────────────────────────────────────────────────────

class _GeminiBackend:
    _BASE = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, key: str):
        self._key = key

    def generate(self, model: Model, prompt: str) -> str:
        url = f"{self._BASE}/{model.id}:generateContent?key={self._key}"
        payload = {
            "system_instruction": {"parts": [{"text": _SYSTEM}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": model.temperature,
                "maxOutputTokens": model.max_tokens,
            },
        }
        data = _http_post(url, payload)
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            raise RuntimeError("Unexpected response format from Gemini.")


class _GroqBackend:
    _URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, key: str):
        self._headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    def generate(self, model: Model, prompt: str) -> str:
        payload = {
            "model": model.id,
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            "temperature": model.temperature,
            "max_tokens": model.max_tokens,
        }
        data = _http_post(self._URL, payload, self._headers)
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            raise RuntimeError("Unexpected response format from Groq.")


# ── Public Client ─────────────────────────────────────────────────────────────

class LLMClient:
    """
    Unified client supporting Gemini and Groq.
    Automatically uses whichever API keys are present.
    """

    def __init__(self):
        g_key = os.getenv("GOOGLE_API_KEY", "").strip()
        q_key = os.getenv("GROQ_API_KEY", "").strip()

        self._backends: Dict[str, object] = {}
        if g_key:
            self._backends["gemini"] = _GeminiBackend(g_key)
        if q_key:
            self._backends["groq"] = _GroqBackend(q_key)

        if not self._backends:
            raise ValueError(
                "No API keys found.\n\n"
                "Add at least one to your .env file:\n"
                "  GOOGLE_API_KEY=...   ← https://aistudio.google.com/app/apikey\n"
                "  GROQ_API_KEY=...     ← https://console.groq.com/keys"
            )

        # Default to first available model whose provider is configured
        self._model = next(
            m for m in _MODELS if m.provider in self._backends
        )

    # ── Public API ──────────────────────────────────────────────────────────

    @property
    def available_models(self) -> List[Model]:
        return [m for m in _MODELS if m.provider in self._backends]

    @property
    def current_model(self) -> Model:
        return self._model

    def set_model(self, model_id: str):
        m = MODEL_MAP.get(model_id)
        if m is None:
            raise ValueError(f"Unknown model: {model_id}")
        if m.provider not in self._backends:
            provider_name = "Google (GOOGLE_API_KEY)" if m.provider == "gemini" else "Groq (GROQ_API_KEY)"
            raise ValueError(f"This model requires {provider_name}, which is not configured.")
        self._model = m

    def provider_status(self) -> Dict[str, bool]:
        return {p: p in self._backends for p in ("gemini", "groq")}

    def generate_mcqs(self, topic: str, n: int = 5,
                      difficulty: str = "medium", choices: int = 4) -> List[Dict]:
        prompt = _build_prompt(topic, n, difficulty, choices)
        backend = self._backends[self._model.provider]
        text = backend.generate(self._model, prompt)
        raw = _parse(text)
        validated = [_validate(mcq, choices) for mcq in raw[:n]]
        result = [m for m in validated if m]
        if not result:
            raise RuntimeError("Model returned no valid questions. Try a different topic or model.")
        return result


# ── .env template ─────────────────────────────────────────────────────────────

def create_env_template() -> str:
    content = (
        "# Get your Google key: https://aistudio.google.com/app/apikey\n"
        "GOOGLE_API_KEY=your_google_api_key_here\n\n"
        "# Get your Groq key: https://console.groq.com/keys\n"
        "GROQ_API_KEY=your_groq_api_key_here\n"
    )
    with open(".env", "w") as f:
        f.write(content)
    return content
