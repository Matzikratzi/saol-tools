from __future__ import annotations

import unittest

from scripts.article_text_review import group_articles, merge_overlapping_rows


def row(page, column, top, text, *, start=False, heading=False):
    return {
        "page": page,
        "column": column,
        "top": float(top),
        "bottom": float(top + 10),
        "text": text,
        "match_text": text.casefold(),
        "left": 100.0,
        "right": 500.0,
        "baseline": start,
        "chapter_heading": heading,
        "ocr_reaches_left": start,
        "pixel_reaches_left": False,
        "features": [0.0] * 18,
    }


class ArticleGroupingTests(unittest.TestCase):
    def test_continuations_follow_start_across_column_and_page(self):
        rows = [
            row(23, 1, 100, "alpha", start=True),
            row(23, 1, 150, "fortsättning"),
            row(23, 2, 80, "över spalt"),
            row(24, 1, 90, "över sida"),
            row(24, 1, 140, "beta", start=True),
        ]
        articles, unattached, headings = group_articles(rows)
        self.assertEqual(len(articles), 2)
        self.assertEqual([line["text"] for line in articles[0]["lines"]], [
            "alpha", "fortsättning", "över spalt", "över sida"
        ])
        self.assertEqual(unattached, [])
        self.assertEqual(headings, [])

    def test_overlapping_tesseract_fragments_become_one_article_start(self):
        suffix = row(29, 2, 494, "-en s. mus. vard. embouchyr", start=True)
        suffix["left"] = 1450.0
        suffix["bottom"] = 548.0
        headword = row(29, 2, 500, "ambis", start=True)
        headword["left"] = 1380.0
        headword["bottom"] = 545.0
        merged = merge_overlapping_rows([suffix, headword])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["text"], "ambis -en s. mus. vard. embouchyr")
        articles, _unattached, _headings = group_articles([suffix, headword])
        self.assertEqual(len(articles), 1)

    def test_heading_is_excluded_and_resets_article(self):
        rows = [
            row(50, 1, 100, "sista a", start=True),
            row(50, 1, 150, "Bb", heading=True),
            row(50, 1, 200, "skräp före första b"),
            row(50, 1, 250, "babian", start=True),
        ]
        articles, unattached, headings = group_articles(rows)
        self.assertEqual(len(articles), 2)
        self.assertEqual(articles[0]["text"], "sista a")
        self.assertEqual([item["text"] for item in unattached], ["skräp före första b"])
        self.assertEqual([item["text"] for item in headings], ["Bb"])


if __name__ == "__main__":
    unittest.main()
