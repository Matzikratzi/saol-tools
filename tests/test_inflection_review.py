from __future__ import annotations

import unittest

from scripts.inflection_review import apply_ending, bracket_variants, parse_inflection


class InflectionReviewTests(unittest.TestCase):
    def test_lodstreck_replaces_the_trailing_headword_part(self):
        self.assertEqual(apply_ending("amp|el", "ampel", "-elt"), ["ampelt"])
        self.assertEqual(apply_ending("amp|el", "ampel", "-la"), ["ampla"])
        self.assertEqual(apply_ending("altar|e", "altare", "-et"), ["altaret"])
        self.assertEqual(apply_ending("ager|a", "agera", "-ade"), ["agerade"])

    def test_suffix_is_appended_without_lodstreck(self):
        self.assertEqual(apply_ending("väg", "väg", "-en"), ["vägen"])
        self.assertEqual(apply_ending("väg", "väg", "-ar"), ["vägar"])
        self.assertEqual(apply_ending("hus", "hus", "="), ["hus"])

    def test_optional_brackets_create_both_variants(self):
        self.assertEqual(bracket_variants("[e]n"), ["n", "en"])
        self.assertEqual(apply_ending("akademi", "akademi", "-[e]n"), [
            "akademin", "akademien"
        ])

    def test_parses_explicit_adjective_forms(self):
        article = {
            "number": 1,
            "start_page": 30,
            "start_column": 1,
            "headword_ocr": "amp|el -elt -la -lare adj. storartad",
        }
        head = {
            "headword": "ampel",
            "stem_headword": "amp|el",
            "homonym": 2,
        }
        result = parse_inflection(article, head)
        self.assertEqual(result["part_of_speech"], "adj")
        self.assertEqual(result["forms"], ["ampel", "ampelt", "ampla", "amplare"])
        self.assertEqual(result["status"], "uttrycklig")


if __name__ == "__main__":
    unittest.main()
