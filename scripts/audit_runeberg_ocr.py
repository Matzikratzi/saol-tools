from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from app.runeberg import (
    _align_lines,
    _normalized_observation_line,
    _observation_line_indices,
    _runeberg_ocr_lines,
    extract_observations,
    ocr_image_url,
    page_urls,
    reconcile_contextual_observations,
)


@dataclass(frozen=True)
class PageResult:
    page: int
    observations: int
    runeberg_values: int
    accepted: int
    conflicts: int
    matched_lines: int
    mean_similarity: float | None


def fetch_with_retries(
    client: httpx.Client,
    url: str,
    *,
    attempts: int,
    timeout: float,
) -> httpx.Response:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = client.get(url, timeout=timeout)
            response.raise_for_status()
            return response
        except (httpx.HTTPError, httpx.TimeoutException) as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(min(2 ** (attempt - 1), 8))
    assert last_error is not None
    raise last_error


def audit_page(
    client: httpx.Client,
    page: int,
    *,
    attempts: int,
    timeout: float,
) -> tuple[PageResult, list[dict[str, object]]]:
    source_url, image_url = page_urls(page)
    source_response = fetch_with_retries(client, source_url, attempts=attempts, timeout=timeout)
    image_response = fetch_with_retries(client, ocr_image_url(image_url), attempts=attempts, timeout=timeout)

    observations = extract_observations(image_response.content)
    runeberg_lines = _runeberg_ocr_lines(source_response.text)
    observation_lines = _observation_line_indices(observations)
    tesseract_lines = [_normalized_observation_line(observations, line) for line in observation_lines]
    runeberg_normalized = [[token.casefold() for token in line] for line in runeberg_lines]
    pairs = _align_lines(tesseract_lines, runeberg_normalized)
    similarities = [score for _, _, score in pairs]

    corrected = reconcile_contextual_observations(observations, source_response.text)
    runeberg_values = [item for item in corrected if item.ocr_runeberg]
    conflicts = [item for item in corrected if item.ocr_conflict]
    accepted = [item for item in runeberg_values if not item.ocr_conflict]

    conflict_rows = [
        {
            "page": page,
            "x": item.left,
            "y": item.top,
            "tesseract": item.ocr_tesseract or "",
            "runeberg": item.ocr_runeberg or "",
            "text": item.text,
        }
        for item in conflicts
    ]
    result = PageResult(
        page=page,
        observations=len(observations),
        runeberg_values=len(runeberg_values),
        accepted=len(accepted),
        conflicts=len(conflicts),
        matched_lines=len(pairs),
        mean_similarity=(sum(similarities) / len(similarities)) if similarities else None,
    )
    return result, conflict_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["page", "x", "y", "tesseract", "runeberg", "text"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Granska OCR-sammanslagningen över ett sidintervall och sammanställ konflikter."
    )
    parser.add_argument("start", nargs="?", type=int, default=19, help="Första sidan, standard 19")
    parser.add_argument("end", nargs="?", type=int, default=100, help="Sista sidan inklusive, standard 100")
    parser.add_argument("--csv", type=Path, help="Skriv samtliga konflikter till CSV")
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP-timeout per hämtning")
    parser.add_argument("--attempts", type=int, default=3, help="Antal hämtningsförsök per URL")
    parser.add_argument("--delay", type=float, default=0.15, help="Paus mellan sidor")
    args = parser.parse_args()

    if args.start < 1 or args.end < args.start:
        parser.error("Sidintervallet måste uppfylla 1 <= start <= slut")
    if args.attempts < 1:
        parser.error("--attempts måste vara minst 1")

    headers = {"User-Agent": "saol-tools/ocr-audit"}
    results: list[PageResult] = []
    conflict_rows: list[dict[str, object]] = []
    failures: list[tuple[int, str]] = []

    with httpx.Client(follow_redirects=True, headers=headers) as client:
        for page in range(args.start, args.end + 1):
            try:
                result, page_conflicts = audit_page(
                    client,
                    page,
                    attempts=args.attempts,
                    timeout=args.timeout,
                )
            except Exception as exc:
                failures.append((page, str(exc)))
                print(f"Sida {page:4d}: FEL {exc}", flush=True)
                continue

            results.append(result)
            conflict_rows.extend(page_conflicts)
            similarity = "-" if result.mean_similarity is None else f"{result.mean_similarity:.3f}"
            print(
                f"Sida {page:4d}: ord={result.observations:4d}, "
                f"Runeberg={result.runeberg_values:3d}, accepterade={result.accepted:3d}, "
                f"konflikter={result.conflicts:3d}, rader={result.matched_lines:3d}, "
                f"likhet={similarity}",
                flush=True,
            )
            if args.delay > 0:
                time.sleep(args.delay)

    if args.csv:
        write_csv(args.csv, conflict_rows)

    total_pages = len(results)
    total_observations = sum(item.observations for item in results)
    total_runeberg = sum(item.runeberg_values for item in results)
    total_accepted = sum(item.accepted for item in results)
    total_conflicts = sum(item.conflicts for item in results)
    pages_with_conflicts = sum(item.conflicts > 0 for item in results)

    print("\n================ SUMMERING ================")
    print(f"Begärt intervall: {args.start}-{args.end}")
    print(f"Bearbetade sidor: {total_pages}")
    print(f"Misslyckade sidor: {len(failures)}")
    print(f"Tesseract-observationer: {total_observations}")
    print(f"Observationer med Runeberg-värde: {total_runeberg}")
    print(f"Automatiskt accepterade: {total_accepted}")
    print(f"Konflikter: {total_conflicts}")
    print(f"Sidor med konflikter: {pages_with_conflicts}")
    if total_runeberg:
        print(f"Andel accepterade Runeberg-värden: {100 * total_accepted / total_runeberg:.2f} %")
        print(f"Andel konflikter bland Runeberg-värden: {100 * total_conflicts / total_runeberg:.2f} %")

    if conflict_rows:
        print("\nKonflikter:")
        for row in conflict_rows:
            print(
                f"  sida {row['page']:4d}, y={row['y']:4d}, x={row['x']:4d}: "
                f"T={row['tesseract']!r}, R={row['runeberg']!r}, text={row['text']!r}"
            )

    if failures:
        print("\nMisslyckade sidor:", file=sys.stderr)
        for page, error in failures:
            print(f"  sida {page}: {error}", file=sys.stderr)

    if args.csv:
        print(f"\nCSV: {args.csv}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
