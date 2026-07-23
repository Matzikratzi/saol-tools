from __future__ import annotations

import unittest

from scripts.article_text_review import group_articles


def row(page, column, top, text, *, start=False, heading=False):
    return {
        "page": page,
        "column": column,
        "top": float(top),
        "bottom": float(top + 10),
        "text": text,
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
