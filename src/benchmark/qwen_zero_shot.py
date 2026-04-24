"""Zero-shot Qwen runner with strict JSON prompting and repair."""

from __future__ import annotations

import ast
import json
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from benchmark.evaluate_model import ModelPredictionBundle
from dataset import VALID_ASPECTS, VALID_SENTIMENTS, coerce_review_id, sanitize_aspect_sentiments


QWEN_HF_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
QWEN_OLLAMA_MODEL_NAME = "qwen2.5:0.5b"


@dataclass
class QwenZeroShotConfig:
    """Configuration for the zero-shot Qwen benchmark runner."""

    provider: str = "auto"
    hf_model_name: str = QWEN_HF_MODEL_NAME
    ollama_model_name: str = QWEN_OLLAMA_MODEL_NAME
    max_new_tokens: int = 200
    temperature: float = 0.0
    top_p: float = 0.95
    retry_invalid_json: bool = True


def normalize_prediction_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Coerce a parsed payload into the expected benchmark schema."""
    aspects = payload.get("aspects", [])
    if isinstance(aspects, str):
        aspects = [aspects]
    if not isinstance(aspects, list):
        aspects = []

    aspect_sentiments = payload.get("aspect_sentiments", {})
    if not isinstance(aspect_sentiments, dict):
        aspect_sentiments = {}

    safe_aspects, safe_sentiments = sanitize_aspect_sentiments(aspects, aspect_sentiments)
    return {
        "aspects": safe_aspects,
        "aspect_sentiments": safe_sentiments,
    }


def extract_json_candidate(text: str) -> str:
    """Extract the most likely JSON object from model output."""
    stripped = str(text or "").strip()
    if stripped.startswith("```"):
        stripped = stripped.replace("```json", "").replace("```JSON", "").replace("```", "").strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        return stripped[start : end + 1]
    return stripped


def parse_and_repair_json(text: str) -> Optional[Dict[str, Any]]:
    """Parse model output into a valid ABSA JSON dict when possible."""
    candidate = extract_json_candidate(text)
    parsers = [json.loads, ast.literal_eval]
    for parser in parsers:
        try:
            parsed = parser(candidate)
        except (json.JSONDecodeError, SyntaxError, ValueError, TypeError):
            continue
        if isinstance(parsed, dict):
            return normalize_prediction_payload(parsed)
    return None


def build_messages(review_text: str, retry_feedback: Optional[str] = None) -> List[Dict[str, str]]:
    """Build a strict JSON prompt for Qwen zero-shot inference."""
    schema_hint = (
        '{"aspects":["service"],"aspect_sentiments":{"service":"negative"}}'
    )
    base_instruction = (
        "You are an Arabic aspect-based sentiment analysis classifier. "
        "Classify the review into zero or more aspects from this exact set: "
        "food, service, price, cleanliness, delivery, ambiance, app_experience, general, none. "
        "For each predicted aspect, assign one sentiment from this exact set: positive, negative, neutral. "
        "If no concrete aspect is present, return aspects=[\"none\"] and aspect_sentiments={\"none\":\"neutral\"}. "
        "Never output any aspect outside the allowed set. "
        "Return valid JSON only with exactly two keys: aspects and aspect_sentiments."
    )
    if retry_feedback:
        base_instruction += " The previous answer was invalid JSON or invalid labels. " + retry_feedback

    return [
        {"role": "system", "content": base_instruction},
        {
            "role": "user",
            "content": (
                f"Review:\n{review_text}\n\n"
                "Output format example:\n"
                f"{schema_hint}\n\n"
                "Return JSON only."
            ),
        },
    ]


def ollama_api_available() -> bool:
    """Return True when the Ollama HTTP API appears reachable."""
    request = urllib.request.Request("http://127.0.0.1:11434/api/tags", method="GET")
    try:
        with urllib.request.urlopen(request, timeout=2) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError):
        return False


def run_ollama_chat(messages: Sequence[Mapping[str, str]], model_name: str, temperature: float) -> str:
    """Call a local Ollama server or CLI and return the raw text response."""
    payload = {
        "model": model_name,
        "messages": list(messages),
        "stream": False,
        "options": {"temperature": temperature},
    }
    if ollama_api_available():
        request = urllib.request.Request(
            "http://127.0.0.1:11434/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            body = json.loads(response.read().decode("utf-8"))
        return str(body.get("message", {}).get("content", ""))

    ollama_binary = shutil.which("ollama")
    if not ollama_binary:
        raise RuntimeError("Ollama is not available via API or CLI.")

    prompt = "\n".join(f"{item['role'].upper()}: {item['content']}" for item in messages)
    completed = subprocess.run(
        [ollama_binary, "run", model_name, prompt],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )
    return completed.stdout.strip()


class TransformersQwenClient:
    """Lazy-loaded local Transformers client for Qwen generation."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def generate(
        self,
        messages: Sequence[Mapping[str, str]],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            encoded = self.tokenizer.apply_chat_template(
                list(messages),
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        else:
            prompt = "\n".join(f"{item['role'].upper()}: {item['content']}" for item in messages)
            encoded = self.tokenizer(prompt, return_tensors="pt").input_ids

        encoded = encoded.to(self.device)
        output = self.model.generate(
            encoded,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0.0,
            temperature=max(temperature, 1e-5),
            top_p=top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        generated = output[0][encoded.shape[-1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


def resolve_provider(config: QwenZeroShotConfig) -> str:
    """Resolve the generation backend used for zero-shot inference."""
    if config.provider in {"ollama", "transformers"}:
        return config.provider
    if ollama_api_available() or shutil.which("ollama"):
        return "ollama"
    return "transformers"


def run_qwen_zero_shot_benchmark(
    validation_df: pd.DataFrame,
    output_dir: Path,
    config: Optional[QwenZeroShotConfig] = None,
) -> ModelPredictionBundle:
    """Run zero-shot Qwen inference on the validation set."""
    cfg = config or QwenZeroShotConfig()
    provider = resolve_provider(cfg)
    transformers_client = TransformersQwenClient(cfg.hf_model_name) if provider == "transformers" else None

    predictions: List[Dict[str, Any]] = []
    retries_used = 0
    invalid_before_retry = 0
    fallbacks_used = 0
    inference_start = time.perf_counter()

    for row in tqdm(validation_df.to_dict(orient="records"), desc="Running Qwen zero-shot", leave=False):
        review_id = row["review_id"]
        review_text = str(row["review_text"])
        raw_response = ""
        parsed_prediction: Optional[Dict[str, Any]] = None

        for attempt in range(2):
            messages = build_messages(
                review_text,
                retry_feedback=(
                    "Return exactly one JSON object with valid labels and no surrounding text."
                    if attempt == 1
                    else None
                ),
            )
            if provider == "ollama":
                try:
                    raw_response = run_ollama_chat(messages, cfg.ollama_model_name, cfg.temperature)
                except Exception:
                    if cfg.provider != "auto":
                        raise
                    provider = "transformers"
                    if transformers_client is None:
                        transformers_client = TransformersQwenClient(cfg.hf_model_name)
                    raw_response = transformers_client.generate(
                        messages,
                        max_new_tokens=cfg.max_new_tokens,
                        temperature=cfg.temperature,
                        top_p=cfg.top_p,
                    )
            else:
                raw_response = transformers_client.generate(
                    messages,
                    max_new_tokens=cfg.max_new_tokens,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                )

            parsed_prediction = parse_and_repair_json(raw_response)
            if parsed_prediction is not None:
                break
            invalid_before_retry += 1
            if attempt == 0 and cfg.retry_invalid_json:
                retries_used += 1
                continue
            break

        if parsed_prediction is None:
            fallbacks_used += 1
            parsed_prediction = {"aspects": ["none"], "aspect_sentiments": {"none": "neutral"}}

        predictions.append(
            {
                "review_id": coerce_review_id(review_id),
                "review_text": review_text,
                "aspects": list(parsed_prediction["aspects"]),
                "aspect_sentiments": dict(parsed_prediction["aspect_sentiments"]),
            }
        )

    inference_seconds = time.perf_counter() - inference_start
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "qwen_runtime_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "provider": provider,
                "config": cfg.__dict__,
                "retries_used": retries_used,
                "invalid_before_retry": invalid_before_retry,
                "fallbacks_used": fallbacks_used,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    return ModelPredictionBundle(
        model_name="qwen_zero_shot",
        model_family="llm_zero_shot",
        predictions=predictions,
        inference_seconds=inference_seconds,
        review_ids=[row["review_id"] for row in validation_df.to_dict(orient="records")],
        aspect_probabilities=None,
        sentiment_probabilities=None,
        thresholds=None,
        training_seconds=None,
        metadata={
            "provider": provider,
            "hf_model_name": cfg.hf_model_name,
            "ollama_model_name": cfg.ollama_model_name,
            "retries_used": retries_used,
            "invalid_before_retry": invalid_before_retry,
            "fallbacks_used": fallbacks_used,
            "runtime_config_path": str(output_dir / "qwen_runtime_config.json"),
        },
    )
