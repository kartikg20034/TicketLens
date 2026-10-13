#!/usr/bin/env python3
"""Count reference-ticket tokens with Gemini and estimate embedding cost."""

from __future__ import annotations

import csv
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
import httpx


load_dotenv(dotenv_path=".env")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_CSV = PROJECT_ROOT / "data/processed/rag_reference.csv"
MODEL = "gemini-embedding-2"
EXPECTED_TICKETS = 6771
PRICE_PER_MILLION_TOKENS_USD = 0.20
REQUEST_TIMEOUT_MS = 30_000
MAX_ATTEMPTS = 5
MAX_WORKERS = 8
PROGRESS_EVERY = 100


def read_reference_texts() -> list[str]:
    with REFERENCE_CSV.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required = {"rag_text", "split"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Reference CSV is missing required columns: {sorted(missing)}")
        texts = []
        for row in reader:
            if row["split"] != "reference":
                raise ValueError("Reference CSV contains a row not marked as reference")
            if not row["rag_text"].strip():
                raise ValueError("Reference CSV contains an empty rag_text")
            texts.append(row["rag_text"])

    if len(texts) != EXPECTED_TICKETS:
        raise ValueError(f"Expected {EXPECTED_TICKETS} reference tickets; found {len(texts)}")
    return texts


def count_one(client: genai.Client, index: int, text: str) -> tuple[int, int]:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = client.models.count_tokens(model=MODEL, contents=text)
            if response.total_tokens is None:
                raise RuntimeError("Gemini countTokens returned no total_tokens value")
            return index, int(response.total_tokens)
        except Exception as error:
            retryable = isinstance(error, httpx.TransportError) or (
                isinstance(error, genai_errors.APIError)
                and error.code in {408, 429, 500, 502, 503, 504}
            )
            if not retryable or attempt == MAX_ATTEMPTS:
                # Do not print exception text: SDK errors can include request data.
                status = getattr(error, "code", None)
                raise RuntimeError(
                    f"Token count failed for ticket index {index + 1}: "
                    f"{type(error).__name__}, status={status}"
                ) from None
            delay = min(2 ** (attempt - 1), 20) + random.uniform(0, 0.5)
            time.sleep(delay)
    raise AssertionError("unreachable")


def estimate(texts: list[str], api_key: str) -> dict[str, float | int]:
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
    )
    counts = [0] * len(texts)
    # Verify the credentials, network, and model's countTokens endpoint once
    # before scheduling thousands of requests.
    _, counts[0] = count_one(client, 0, texts[0])
    completed = 1
    print(f"Token-count progress: {completed}/{len(texts)} tickets; errors=0")
    failures: list[str] = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(count_one, client, index, text): index
            for index, text in enumerate(texts[1:], start=1)
        }
        for future in as_completed(futures):
            try:
                index, token_count = future.result()
                counts[index] = token_count
            except Exception as error:
                failures.append(str(error))
            completed += 1
            if completed % PROGRESS_EVERY == 0 or completed == len(texts):
                print(f"Token-count progress: {completed}/{len(texts)} tickets; errors={len(failures)}")

    if failures:
        preview = ", ".join(failures[:5])
        raise RuntimeError(
            f"Token counting failed for {len(failures)} ticket(s); first errors: {preview}. "
            "Rerun after resolving the API issue."
        )

    total_tokens = sum(counts)
    return {
        "number_of_tickets": len(counts),
        "total_input_tokens": total_tokens,
        "average_tokens_per_ticket": total_tokens / len(counts),
        "minimum_tokens": min(counts),
        "maximum_tokens": max(counts),
        "estimated_cost_usd": total_tokens / 1_000_000 * PRICE_PER_MILLION_TOKENS_USD,
    }


def main() -> int:
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        print("GEMINI_API_KEY is not set in the environment or project .env file.")
        return 1
    try:
        report = estimate(read_reference_texts(), api_key)
        print(f"Number of tickets: {report['number_of_tickets']:,}")
        print(f"Total input tokens: {report['total_input_tokens']:,}")
        print(f"Average tokens per ticket: {report['average_tokens_per_ticket']:.2f}")
        print(f"Minimum tokens: {report['minimum_tokens']:,}")
        print(f"Maximum tokens: {report['maximum_tokens']:,}")
        print(f"Estimated Gemini Embedding 2 cost: ${report['estimated_cost_usd']:.6f} USD")
        print(f"Estimated cost for all {report['number_of_tickets']:,} reference tickets: ${report['estimated_cost_usd']:.6f} USD")
        return 0
    except Exception as error:
        print(f"Cost estimate failed: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
