from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "app" / "runeberg.py"

OLD = '''    tesseract_keys = [" ".join(line) for line in tesseract_lines]
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
'''

NEW = '''    corrected = list(observations)

    # Align physical lines monotonically with dynamic programming. A skipped
    # line receives a modest penalty, while a plausible line pair receives a
    # score based on token similarity. This prevents one missing/split line
    # from shifting all following lines.
    left_count = len(tesseract_lines)
    right_count = len(runeberg_normalized)
    gap_penalty = -0.28
    minimum_pair_similarity = 0.24

    scores = [[0.0] * (right_count + 1) for _ in range(left_count + 1)]
    moves = [[""] * (right_count + 1) for _ in range(left_count + 1)]

    for left_index in range(1, left_count + 1):
        scores[left_index][0] = left_index * gap_penalty
        moves[left_index][0] = "skip_left"
    for right_index in range(1, right_count + 1):
        scores[0][right_index] = right_index * gap_penalty
        moves[0][right_index] = "skip_right"

    for left_index in range(1, left_count + 1):
        for right_index in range(1, right_count + 1):
            similarity = _line_similarity(
                tesseract_lines[left_index - 1],
                runeberg_normalized[right_index - 1],
            )
            pair_score = scores[left_index - 1][right_index - 1] + (similarity - 0.18)
            skip_left_score = scores[left_index - 1][right_index] + gap_penalty
            skip_right_score = scores[left_index][right_index - 1] + gap_penalty

            best_score = max(pair_score, skip_left_score, skip_right_score)
            scores[left_index][right_index] = best_score
            if best_score == pair_score:
                moves[left_index][right_index] = "pair"
            elif best_score == skip_left_score:
                moves[left_index][right_index] = "skip_left"
            else:
                moves[left_index][right_index] = "skip_right"

    line_pairs: list[tuple[int, int]] = []
    left_index = left_count
    right_index = right_count
    while left_index > 0 or right_index > 0:
        move = moves[left_index][right_index]
        if move == "pair":
            candidate_left = left_index - 1
            candidate_right = right_index - 1
            similarity = _line_similarity(
                tesseract_lines[candidate_left],
                runeberg_normalized[candidate_right],
            )
            if similarity >= minimum_pair_similarity:
                line_pairs.append((candidate_left, candidate_right))
            left_index -= 1
            right_index -= 1
        elif move == "skip_left":
            left_index -= 1
        elif move == "skip_right":
            right_index -= 1
        else:
            break

    line_pairs.reverse()
'''


def main() -> None:
    source = TARGET.read_text(encoding="utf-8")
    if NEW in source:
        print("Dynamisk radalignering finns redan i app/runeberg.py")
        return
    if OLD not in source:
        raise SystemExit("Hittade inte den tidigare radaligneringen i app/runeberg.py")
    TARGET.write_text(source.replace(OLD, NEW, 1), encoding="utf-8")
    print("Uppdaterade app/runeberg.py med dynamisk radalignering")


if __name__ == "__main__":
    main()
