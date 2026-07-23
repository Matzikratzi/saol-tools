from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.article_start_ml import (
    FEATURE_NAMES,
    _slanted_geometry,
    align_truth,
    compare_models,
    normalize_word,
    read_ground_truth,
)


class FakeLine:
    def __init__(self, y: float, x: float):
        self.top = y
        self.bottom = y + 50
        self.raw_start_x = x


class GroundTruthTests(unittest.TestCase):
    def test_normalize_lodstreck_and_punctuation(self):
        self.assertEqual(normalize_word("acklimatiser|a"), "acklimatisera")
        self.assertEqual(normalize_word("A-dur"), "adur")

    def test_read_both_supported_page_heading_forms(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "truth.txt"
            path.write_text(
                "sida: 19\nvänster:\na (homonym 1)\nhöger:\nabscess\n"
                "sida 20:\nvänster:\nabsens\nhöger:\naccedera\n",
                encoding="utf-8",
            )
            truth = read_ground_truth(path)
        self.assertEqual(truth[19][1], ["a"])
        self.assertEqual(truth[20][2], ["accedera"])

    def test_alignment_is_ordered_and_tolerates_ocr_errors(self):
        rows = [
            {"match_text": "fortsättning"},
            {"match_text": "abskissla skiss"},
            {"match_text": "mera text"},
            {"match_text": "absoiut"},
        ]
        matches = align_truth(["abskissa", "absolut"], rows)
        self.assertEqual([index for index, _word, _score in matches], [1, 3])
        self.assertTrue(all(score > 0.7 for _index, _word, score in matches))

    def test_models_use_leave_one_page_out_predictions(self):
        rows = []
        for page in (19, 20, 21, 22):
            for index in range(12):
                label = int(index % 3 == 0)
                features = [0.0] * len(FEATURE_NAMES)
                features[2] = 0.10 if label else 0.55
                features[8] = 0.90 if label else 0.10
                features[12] = -0.8 if label else 0.8
                rows.append(
                    {
                        "page": page,
                        "column": 1,
                        "top": float(index * 50),
                        "text": f"rad {page} {index}",
                        "facit_word": "ord" if label else "",
                        "label": label,
                        "baseline": bool(label),
                        "features": features,
                    }
                )
        results, details = compare_models(rows)
        self.assertEqual(len(details), len(rows))
        self.assertEqual(results["T-regel"]["f1"], 1.0)
        self.assertGreater(results["Logistisk regression"]["f1"], 0.9)
        self.assertGreater(results["Gradient boosting"]["f1"], 0.9)

    def test_slanted_geometry_recovers_parallel_a_and_f_levels(self):
        lines = []
        for index in range(30):
            y = 200.0 + index * 70.0
            base = 160.0 if index % 3 == 0 else 200.0
            lines.append(FakeLine(y, base + 0.02 * y))
        slope, _anchor, haf = _slanted_geometry(lines, 50.0, None)
        self.assertAlmostEqual(slope, 0.02, places=3)
        self.assertIsNotNone(haf)
        self.assertAlmostEqual(haf[2] - haf[1], 40.0, places=1)


if __name__ == "__main__":
    unittest.main()
