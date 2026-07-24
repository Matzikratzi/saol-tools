from __future__ import annotations

import unittest

from scripts.headword_review import (
    extract_head,
    fits_local_alphabetic_window,
    infer_homonym_runs,
    infer_stem_boundary_from_ocr,
    reconcile_homonym_neighbours,
    recover_short_homonym_run,
    repair_alphabetic_accents,
    swedish_sort_key,
    trim_plain_definition_tails_by_order,
    visibly_lighter_than_head,
)
from scripts.runeberg_headwords import align_lines, raw_headword


class HeadwordReviewTests(unittest.TestCase):
    def test_definition_preposition_ends_hyphenated_head(self):
        article = {
            "lines": [{"tokens": [
                {"text": "A-aktie", "left": 100},
                {"text": "av", "left": 250},
                {"text": "serie", "left": 310},
            ]}]
        }
        self.assertEqual(extract_head(article)["headword"], "a-aktie")

    def test_three_short_entries_restore_lost_homonym_digits(self):
        items = []
        for headword, runeberg, marker in [
            ("a a-t a-et", "a a-t", True),
            ("à", "à", False),
            ("a", "a", False),
        ]:
            items.append({
                "headword": headword, "raw_headword": headword,
                "stem_headword": headword,
                "runeberg_headword": runeberg,
                "homonym": None, "homonym_inferred": False,
                "homonym_marker_detected": marker,
                "reasons": ["homonymtecknet"] if marker else [],
                "status": "osäker" if marker else "preliminär",
                "corrected_from": "", "correction_method": "",
            })
        recover_short_homonym_run(items)
        self.assertEqual(
            [(item["homonym"], item["headword"]) for item in items],
            [(1, "a"), (2, "a"), (3, "a")],
        )

    def test_plain_a_kassa_definition_is_trimmed_by_style_and_order(self):
        article = {
            "number": 59,
            "lines": [
                {
                    "tokens": [
                        {
                            "text": "A-kassa",
                            "left": 1413,
                            "ink_density": 0.468,
                        },
                        {
                            "text": "arbetslöshetskassa",
                            "left": 1673,
                            "ink_density": 0.358,
                        },
                    ]
                }
            ],
        }
        items = [
            {
                "article_number": 58,
                "headword": "akantus",
                "corrected_from": "",
                "correction_method": "",
            },
            {
                "article_number": 59,
                "headword": "a-kassa arbetslöshetskassa",
                "raw_headword": "A-kassa arbetslöshetskassa",
                "stem_headword": "A-kassa arbetslöshetskassa",
                "corrected_from": "",
                "correction_method": "",
            },
            {
                "article_number": 60,
                "headword": "akatalektisk",
                "corrected_from": "",
                "correction_method": "",
            },
        ]
        self.assertTrue(
            visibly_lighter_than_head(
                article["lines"][0]["tokens"][:1],
                article["lines"][0]["tokens"][1],
            )
        )
        trim_plain_definition_tails_by_order(items, [article])
        self.assertEqual(items[1]["headword"], "a-kassa")
        self.assertEqual(
            items[1]["correction_method"],
            "fetstil och alfabetisk ordning",
        )

    def test_three_neighbours_tolerate_one_bad_immediate_head(self):
        items = [
            {"headword": "akademi"},
            {"headword": "akajer"},
            {"headword": "zz-fel"},
            {"headword": "a-kassa arbetslöshetskassa"},
            {"headword": "aa-fel"},
            {"headword": "akatalektisk"},
            {"headword": "akleja"},
        ]
        self.assertTrue(
            fits_local_alphabetic_window(items, 3, "a-kassa")
        )
        self.assertFalse(
            fits_local_alphabetic_window(
                items, 3, "arbetslöshetskassa"
            )
        )

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
