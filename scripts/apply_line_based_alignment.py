from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "app" / "runeberg.py"

NEW_CODE = r'''def _runeberg_ocr_lines(html: str) -> list[list[str]]:
    """Return Runeberg OCR as physical lines, preserving the page's <br> breaks."""
    lines: list[list[str]] = []
    for raw_line in _runeberg_ocr_text(html).splitlines():
        tokens: list[str] = []
        for raw in raw_line.split():
            token = raw.strip(".,:;!?()[]{}<>\"“”")
            if _word_letters(token):
                tokens.append(token)
        if tokens:
            lines.append(tokens)
    return lines


def _observation_line_indices(observations: list[WordObservation]) -> list[list[int]]:
    """Group Tesseract words into physical lines, left column before right."""
    if not observations:
        return []

    page_left = min(item.left for item in observations)
    page_right = max(item.left + item.width for item in observations)
    split = page_left + (page_right - page_left) / 2
    median_height = sorted(item.height for item in observations)[len(observations) // 2]
    tolerance = max(3, int(median_height * 0.60))

    columns = [
        [index for index, item in enumerate(observations) if item.left < split],
        [index for index, item in enumerate(observations) if item.left >= split],
    ]
    result: list[list[int]] = []
    for column in columns:
        ordered = sorted(column, key=lambda index: (observations[index].top, observations[index].left))
        grouped: list[list[int]] = []
        line_centers: list[float] = []
        for index in ordered:
            item = observations[index]
            center = item.top + item.height / 2
            if grouped and abs(center - line_centers[-1]) <= tolerance:
                grouped[-1].append(index)
                line_centers[-1] = sum(
                    observations[i].top + observations[i].height / 2 for i in grouped[-1]
                ) / len(grouped[-1])
            else:
                grouped.append([index])
                line_centers.append(center)
        for line in grouped:
            result.append(sorted(line, key=lambda index: observations[index].left))
    return result


def _normalized_observation_line(observations: list[WordObservation], indices: list[int]) -> list[str]:
    return [
        normalized
        for index in indices
        if (normalized := _word_letters(observations[index].text))
    ]


def _line_similarity(left: list[str], right: list[str]) -> float:
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right, autojunk=False).ratio()


def reconcile_contextual_observations(
    observations: list[WordObservation],
    runeberg_lines: list[list[str]],
) -> list[WordObservation]:
    """Align OCR by physical line first, then compare tokens within each line."""
    observation_lines = _observation_line_indices(observations)
    if not observation_lines or not runeberg_lines:
        return observations

    tesseract_lines = [
        _normalized_observation_line(observations, indices)
        for indices in observation_lines
    ]
    runeberg_normalized = [
        [_word_letters(token) for token in line if _word_letters(token)]
        for line in runeberg_lines
    ]
    tesseract_keys = [" ".join(line) for line in tesseract_lines]
    runeberg_keys = [" ".join(line) for line in runeberg_normalized]

    corrected = list(observations)
    line_matcher = SequenceMatcher(None, tesseract_keys, runeberg_keys, autojunk=False)
    line_pairs: list[tuple[int, int]] = []

    for tag, left_start, left_end, right_start, right_end in line_matcher.get_opcodes():
        if tag == "equal":
            line_pairs.extend(zip(range(left_start, left_end), range(right_start, right_end)))
            continue
        if tag != "replace":
            continue

        left_count = left_end - left_start
        right_count = right_end - right_start
        if left_count != right_count:
            continue
        for offset in range(left_count):
            left_index = left_start + offset
            right_index = right_start + offset
            if _line_similarity(tesseract_lines[left_index], runeberg_normalized[right_index]) >= 0.35:
                line_pairs.append((left_index, right_index))

    for observation_line_index, runeberg_line_index in line_pairs:
        observation_indices = observation_lines[observation_line_index]
        left_tokens = [_word_letters(observations[index].text) for index in observation_indices]
        right_tokens = runeberg_normalized[runeberg_line_index]
        right_original = runeberg_lines[runeberg_line_index]
        token_matcher = SequenceMatcher(None, left_tokens, right_tokens, autojunk=False)

        for tag, left_start, left_end, right_start, right_end in token_matcher.get_opcodes():
            if tag != "replace":
                continue
            if left_end - left_start != 1 or right_end - right_start != 1:
                continue
            _apply_contextual_replacement(
                corrected,
                observation_indices[left_start],
                right_original[right_start],
                right_tokens[right_start],
            )

    return corrected
'''


def main() -> None:
    source = TARGET.read_text(encoding="utf-8")

    if "def _runeberg_ocr_lines(" in source and "Align OCR by physical line first" in source:
        print("Radbaserad alignering finns redan i app/runeberg.py")
        return

    pattern = re.compile(
        r"def reconcile_contextual_observations\(.*?\n(?=def reconcile_stem_marked_observations\()",
        flags=re.DOTALL,
    )
    match = pattern.search(source)
    if not match:
        raise SystemExit("Hittade inte reconcile_contextual_observations i app/runeberg.py")

    source = source[:match.start()] + NEW_CODE + "\n\n" + source[match.end():]

    old_fetch = """        runeberg_tokens = _runeberg_ocr_tokens(source_response.text)\n        observations = reconcile_contextual_observations(observations, runeberg_tokens)\n"""
    new_fetch = """        runeberg_tokens = _runeberg_ocr_tokens(source_response.text)\n        runeberg_lines = _runeberg_ocr_lines(source_response.text)\n        observations = reconcile_contextual_observations(observations, runeberg_lines)\n"""
    if old_fetch not in source:
        raise SystemExit("Hittade inte anropet till reconcile_contextual_observations")
    source = source.replace(old_fetch, new_fetch, 1)

    TARGET.write_text(source, encoding="utf-8")
    print("Uppdaterade app/runeberg.py med radbaserad alignering")


if __name__ == "__main__":
    main()
