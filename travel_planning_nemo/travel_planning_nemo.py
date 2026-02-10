import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import httpx
from dotenv import load_dotenv

TEMPLATE_PATH = Path(__file__).parent / "templates" / "itinerary_template_v1.md"


@dataclass(frozen=True)
class NvidiaChatConfig:
    base_url: str
    api_key: str
    model_name: str
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 2048

    @staticmethod
    def from_env() -> "NvidiaChatConfig":
        load_dotenv()

        base_url = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1").rstrip("/")
        api_key = os.getenv("NVIDIA_API_KEY", "")
        model_name = os.getenv("MODEL_NAME", "meta/llama-3.1-70b-instruct")

        if not api_key:
            raise RuntimeError("Missing NVIDIA_API_KEY. Create .env and set NVIDIA_API_KEY.")

        def _f(name: str, default: float) -> float:
            v = os.getenv(name)
            return float(v) if v else default

        def _i(name: str, default: int) -> int:
            v = os.getenv(name)
            return int(v) if v else default

        return NvidiaChatConfig(
            base_url=base_url,
            api_key=api_key,
            model_name=model_name,
            temperature=_f("TEMPERATURE", 0.7),
            top_p=_f("TOP_P", 0.9),
            max_tokens=_i("MAX_TOKENS", 2048),
        )


def load_template_text() -> str:
    return TEMPLATE_PATH.read_text(encoding="utf-8")


def render_template(template: str, values: Dict[str, Any]) -> str:
    out = template
    for k, v in values.items():
        out = out.replace(f"{{{{{k}}}}}", str(v))
    return out


def build_user_prompt(trip_request: Dict[str, Any]) -> str:
    template = load_template_text()

    # Flight integration fields (may be filled by agent after calling flight_search)
    flight_context = trip_request.get("flight_context_markdown") or "Flight details not provided."
    arrival_window = trip_request.get("arrival_window") or "Not specified"
    departure_window = trip_request.get("departure_window") or "Not specified"

    values = {
        "destination": trip_request.get("destination", "UNKNOWN"),
        "start_date": trip_request.get("start_date", "UNKNOWN"),
        "end_date": trip_request.get("end_date", "UNKNOWN"),
        "travelers": trip_request.get("travelers", "Not specified"),
        "budget": trip_request.get("budget", "Not specified"),
        "travel_style": trip_request.get("travel_style", "Balanced"),
        "interests": trip_request.get("interests", "General sightseeing"),
        "day_start_time": trip_request.get("day_start_time", "09:00"),
        "pace": trip_request.get("pace", "Moderate"),
        "mobility": trip_request.get("mobility", "No constraints"),
        "food_prefs": trip_request.get("food_prefs", "No constraints"),
        "flight_context": flight_context,
        "arrival_window": arrival_window,
        "departure_window": departure_window,
    }

    filled_template = render_template(template, values)

    constraints = trip_request.get("constraints", "")
    special = trip_request.get("special_requests", "")

    return (
        "Generate a complete itinerary using the template below.\n\n"
        "Hard requirements:\n"
        "- Follow the template headings exactly.\n"
        "- Include 2 optional swaps per day.\n"
        "- Add realistic transit notes and cost ranges.\n"
        "- Output only Markdown.\n"
        "- Do not mention that you are using a template.\n\n"
        "Integration rules:\n"
        "- If Flight Context is provided, align Day 1 and last day with the arrival/departure assumptions.\n"
        "- Do not invent exact flight prices or guaranteed schedules.\n\n"
        f"Extra constraints (if any): {constraints}\n"
        f"Special requests (if any): {special}\n\n"
        "TEMPLATE (filled inputs):\n"
        "-------------------------\n"
        f"{filled_template}\n"
    )


async def call_nvidia_chat_completion(
    *,
    cfg: NvidiaChatConfig,
    system_prompt: str,
    user_prompt: str,
) -> str:
    url = f"{cfg.base_url}/chat/completions"

    headers = {
        "Authorization": f"Bearer {cfg.api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    payload = {
        "model": cfg.model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "max_tokens": cfg.max_tokens,
        "stream": False,
        # Helps prevent ReAct artifacts if the model tries to emit them
        "stop": ["\n\nObservation:", "\n\nAction:", "\n\nThought:"],
    }

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

    return data["choices"][0]["message"]["content"]
