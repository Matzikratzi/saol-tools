from __future__ import annotations

import unittest

from scripts.headword_review import (
    infer_homonym_runs,
    infer_stem_boundary_from_ocr,
    reconcile_homonym_neighbours,
    repair_alphabetic_accents,
    swedish_sort_key,
)
from scripts.runeberg_headwords import align_lines, raw_headword


class HeadwordReviewTests(unittest.TestCase):
    def test_swedish_sort_places_a_accent_with_a_and_aa_after_z(self):
        self.assertLess(swedish_sort_key("à la carte"), swedish_sort_key("aladåb"))
        self.assertGreater(swedish_sort_key("å la carte"), swedish_sort_key("aladåb"))

    def test_extracts_headword_from_runeberg_stem_boundary(self):
        self.assertEqual(raw_headword("amöb|a (-ö'-) -an -or s."), "amöba")
        self.assertEqual(raw_headword("alabaster (-ast'-) -n s."), "alabaster")
        self.assertEqual(raw_headword("^aga ei. åga s. i uttr."), "aga")
        self.assertEqual(
            raw_headword(
                "aggiutination -en -ers. agglutiner|a",
                preserve_boundaries=True,
            ),
            "aggiutination",
        )

    def test_monotonic_secondary_alignment_uses_rest_of_line(self):
        items = [
            {"source_line": "(-ast'-) -n s. vit finkornig o."},
            {"source_line": "å la bonne heure (allabånör)"},
        ]
        lines = [
            "al -en -ar s.",
            "alabaster (-ast'-) -n s. vit finkornig o.",
            "genomskinlig gips",
            "à la bonne heure (allabånör)",
        ]
        matches = align_lines(items, lines)
        self.assertEqual([index for index, _score in matches], [1, 3])
        self.assertTrue(all(score > 0.7 for _index, score in matches))

    def test_l_like_glyph_becomes_stem_boundary_from_canonical_homonym(self):
        self.assertEqual(
            infer_stem_boundary_from_ocr("amploel amplel", "ampel"),
            "amp|el",
        )

    def test_second_homonym_repairs_and_numbers_its_neighbour(self):
        items = [
            {
                "headword": "ampel",
                "homonym": None,
                "homonym_marker_detected": True,
                "runeberg_headword": "kärleks-ampel",
                "reasons": ["homonymtecknet känns igen som citattecken"],
                "status": "osäker",
                "corrected_from": "",
                "correction_method": "",
                "homonym_inferred": False,
            },
            {
                "headword": "amploel amplel",
                "homonym": 2,
                "homonym_marker_detected": True,
                "runeberg_headword": "ampel",
                "runeberg_stem_headword": "amp|el",
                "stem_headword": "amploel amplel",
                "reasons": ["låg OCR-säkerhet (50)"],
                "status": "osäker",
                "corrected_from": "",
                "correction_method": "",
                "homonym_inferred": False,
            },
        ]
        reconcile_homonym_neighbours(items)
        infer_homonym_runs(items)
        self.assertEqual([item["headword"] for item in items], ["ampel", "ampel"])
        self.assertEqual([item["homonym"] for item in items], [1, 2])
        self.assertEqual(items[1]["stem_headword"], "amp|el")
        self.assertEqual(items[1]["correction_method"], "angränsande homonym")
        self.assertTrue(items[0]["homonym_inferred"])

    def test_repairs_aa_to_grave_accent_when_neighbours_prove_it(self):
        items = [
            {"headword": "al"},
            {"headword": "å la bonne heure"},
            {"headword": "å la carte"},
            {"headword": "aladåb"},
        ]
        for item in items:
            item["corrected_from"] = ""
            item["correction_method"] = ""
        repair_alphabetic_accents(items)
        self.assertEqual(items[1]["headword"], "à la bonne heure")
        self.assertEqual(items[2]["headword"], "à la carte")
        self.assertEqual(items[1]["correction_method"], "alfabetisk ordning")


if __name__ == "__main__":
    unittest.main()
