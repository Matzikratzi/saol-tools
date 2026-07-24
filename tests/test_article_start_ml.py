from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.article_start_ml import (
    FEATURE_NAMES,
    _is_chapter_heading,
    _ocr_band_ranges,
    _slanted_geometry,
    align_truth,
    compare_models,
    compare_t_positions,
    normalize_word,
    read_ground_truth,
)


class FakeLine:
    def __init__(self, y: float, x: float):
        self.top = y
        self.bottom = y + 50
        self.raw_start_x = x


class OcrBandTests(unittest.TestCase):
    def test_padded_bands_cover_height_once_by_their_cores(self):
        bands = _ocr_band_ranges(1150, core_height=480, padding=100)
        self.assertEqual(
            bands,
            [
                (0, 580, 0, 480),
                (380, 1060, 480, 960),
                (860, 1150, 960, 1150),
            ],
        )
        self.assertEqual(bands[0][2], 0)
        self.assertEqual(bands[-1][3], 1150)
        for previous, current in zip(bands, bands[1:]):
            self.assertEqual(previous[3], current[2])
            self.assertLess(current[0], current[2])
            self.assertGreater(previous[1], previous[3])


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

    def test_alignment_prefers_left_ink_over_similar_continuation_word(self):
        rows = [
            {
                "match_text": "akt s 1 et er",
                "pixel_reaches_left": True,
                "ocr_reaches_left": False,
            },
            {
                "match_text": "orgel abstraktion ksjön",
                "pixel_reaches_left": False,
                "ocr_reaches_left": False,
            },
        ]
        matches = align_truth(["abstrakt"], rows)
        self.assertEqual(matches[0][0], 0)

    def test_t_position_sweep_reports_each_fraction(self):
        rows = [
            {
                "label": 1,
                "baseline": True,
                "usable": True,
                "t_candidates": {"0.35": False, "0.50": True, "0.65": True},
            },
            {
                "label": 0,
                "baseline": False,
                "usable": True,
                "t_candidates": {"0.35": False, "0.50": False, "0.65": True},
            },
        ]
        results = compare_t_positions(rows)
        self.assertEqual(results["0.50"]["f1"], 1.0)
        self.assertEqual(results["0.35"]["fn"], 1)
        self.assertEqual(results["0.65"]["fp"], 1)

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

    def test_large_uppercase_lowercase_pair_is_chapter_heading(self):
        self.assertTrue(_is_chapter_heading("Aa", 100.0, 50.0))
        self.assertTrue(_is_chapter_heading("Öö", 90.0, 50.0))
        self.assertFalse(_is_chapter_heading("Aa", 60.0, 50.0))
        self.assertFalse(_is_chapter_heading("Ab", 100.0, 50.0))


if __name__ == "__main__":
    unittest.main()
