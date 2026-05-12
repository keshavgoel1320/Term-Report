"""Vertex AI LLM client with runtime model discovery and caching."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model discovery candidates — ordered by preference
# ---------------------------------------------------------------------------

GENERATION_CANDIDATES: tuple[str, ...] = (
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
)

JUDGE_CANDIDATES: tuple[str, ...] = (
    "gemini-2.5-flash-lite",
)

LATEST_MODEL_DOCS = "https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models"


class SelectedModels(BaseModel):
    """The models selected at runtime after probing."""

    generation_model: str
    judge_model: str


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------

class VertexLLMClient:
    """Vertex AI Gemini client with caching and JSON parsing."""

    def __init__(
        self,
        *,
        project: str | None = None,
        location: str = "us-central1",
        cache_dir: Path | None = None,
    ) -> None:
        self._project = project or os.environ.get("GOOGLE_CLOUD_PROJECT", os.environ.get("CLOUD_ML_PROJECT_ID"))
        self._location = location
        self._cache_dir = cache_dir
        self._client: Any = None
        self._selected_models: SelectedModels | None = None

    # ---------- initialisation ----------

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from google import genai
            self._client = genai.Client(
                vertexai=True,
                project=self._project,
                location=self._location,
            )
            logger.info("Vertex AI genai client initialised (project=%s, location=%s)", self._project, self._location)
        except Exception as exc:
            raise RuntimeError(f"Failed to initialise Vertex AI client: {exc}") from exc
        return self._client

    # ---------- model probing ----------

    def discover_models(self) -> SelectedModels:
        """Probe candidates for availability and return the best pair."""
        if self._selected_models is not None:
            return self._selected_models

        client = self._ensure_client()
        generation_model = self._probe_first_available(client, GENERATION_CANDIDATES, "generation")
        judge_model = self._probe_first_available(client, JUDGE_CANDIDATES, "judge")
        self._selected_models = SelectedModels(generation_model=generation_model, judge_model=judge_model)
        logger.info("Selected models: generation=%s, judge=%s", generation_model, judge_model)
        return self._selected_models

    def _probe_first_available(self, client: Any, candidates: tuple[str, ...], role: str) -> str:
        for model_name in candidates:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents="Respond with exactly: OK",
                    config={"max_output_tokens": 10, "temperature": 0.0},
                )
                if response and response.text:
                    logger.info("Model probe OK for %s role: %s", role, model_name)
                    return model_name
            except Exception as exc:
                logger.debug("Model probe failed for %s: %s (%s)", model_name, type(exc).__name__, exc)
                continue
        raise RuntimeError(f"No available model found for {role} role. Tried: {candidates}")

    # ---------- generation ----------

    def generate_text(
        self,
        prompt: str,
        *,
        model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> str:
        """Generate text from a prompt."""
        client = self._ensure_client()
        models = self.discover_models()
        model = model or models.generation_model

        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            },
        )
        return response.text.strip() if response and response.text else ""

    def generate_json(
        self,
        prompt: str,
        *,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 8192,
        cache_key: str | None = None,
    ) -> Any:
        """Generate JSON from a prompt, with optional disk caching."""
        # Check cache first
        if cache_key and self._cache_dir:
            cached = self._read_cache(cache_key)
            if cached is not None:
                return cached

        client = self._ensure_client()
        models = self.discover_models()
        model = model or models.generation_model

        # Add JSON instruction
        full_prompt = prompt + "\n\nIMPORTANT: Return ONLY valid JSON. No markdown fences, no explanation."

        last_error: Exception | None = None
        for attempt in range(5):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=full_prompt,
                    config={
                        "temperature": temperature,
                        "max_output_tokens": max_tokens,
                        "response_mime_type": "application/json",
                    },
                )
                raw_text = response.text.strip() if response and response.text else ""
                if not raw_text:
                    raise ValueError("Empty response from model")

                parsed = self._parse_json(raw_text)

                # Cache the result
                if cache_key and self._cache_dir:
                    self._write_cache(cache_key, parsed)

                return parsed

            except Exception as exc:
                last_error = exc
                err_str = str(exc)
                # Exponential backoff, longer for rate limits
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    wait = min(30, 5 * (2 ** attempt))
                    logger.warning("Rate limited (attempt %d), waiting %ds...", attempt + 1, wait)
                else:
                    wait = 1.0 * (attempt + 1)
                    logger.warning("JSON generation attempt %d failed: %s", attempt + 1, exc)
                if attempt < 4:
                    time.sleep(wait)

        raise RuntimeError(f"JSON generation failed after 5 attempts: {last_error}")

    # ---------- JSON parsing ----------

    @staticmethod
    def _parse_json(text: str) -> Any:
        """Parse JSON, stripping markdown fences and invisible characters."""
        # Strip BOM, null bytes, and other invisible characters
        cleaned = text.strip()
        cleaned = cleaned.lstrip("\ufeff\u200b\u200c\u200d\u2060")
        cleaned = cleaned.replace("\x00", "")

        # Strip markdown fences
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = lines[1:]  # drop opening fence
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        # Direct parse
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON array or object with greedy match
        for pattern in [r'\[[\s\S]*\]', r'\{[\s\S]*\}']:
            match = re.search(pattern, cleaned)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    continue

        # Last resort: try fixing common issues (trailing commas, etc.)
        fixed = re.sub(r',\s*([}\]])', r'\1', cleaned)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

        raise ValueError(f"Could not parse JSON from response: {cleaned[:200]}...")

    # ---------- disk cache ----------

    def _cache_path(self, key: str) -> Path:
        assert self._cache_dir is not None
        hashed = hashlib.sha256(key.encode()).hexdigest()[:16]
        safe_key = re.sub(r'[^\w\-]', '_', key)[:60]
        return self._cache_dir / f"{safe_key}_{hashed}.json"

    def _read_cache(self, key: str) -> Any | None:
        path = self._cache_path(key)
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return None
        return None

    def _write_cache(self, key: str, data: Any) -> None:
        assert self._cache_dir is not None
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._cache_path(key)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
