from __future__ import annotations

import unittest

from scripts.lemma_rule_audit import facit_deviations


class LemmaRuleAuditTests(unittest.TestCase):
    def test_facit_deviations_separate_unexpected_and_missing(self):
        items = [
            {
                "article_number": 1,
                "lemma": "rätt",
                "source_page": 23,
                "source_column": 1,
                "source_top": 100,
                "source_left": 100,
            },
            {
                "article_number": 1,
                "lemma": "extra",
                "source_page": 23,
                "source_column": 1,
                "source_top": 110,
                "source_left": 100,
            },
        ]
        facit = {
            "pages": {
                "23": {
                    "candidates": [
                        {"article_number": 1, "lemma": "rätt"},
                        {"article_number": 1, "lemma": "saknas"},
                    ]
                }
            }
        }
        self.assertEqual(facit_deviations(items, facit), (1, 1))


if __name__ == "__main__":
    unittest.main()
