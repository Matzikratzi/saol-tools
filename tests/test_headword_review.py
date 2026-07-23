from __future__ import annotations

import unittest

from scripts.headword_review import repair_alphabetic_accents, swedish_sort_key


class HeadwordReviewTests(unittest.TestCase):
    def test_swedish_sort_places_a_accent_with_a_and_aa_after_z(self):
        self.assertLess(swedish_sort_key("à la carte"), swedish_sort_key("aladåb"))
        self.assertGreater(swedish_sort_key("å la carte"), swedish_sort_key("aladåb"))

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
