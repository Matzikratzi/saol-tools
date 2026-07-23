from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.lemma_review import (
    _items_by_printed_row,
    _items_in_reading_order,
    expand_compound,
    infer_compound_series_boundary,
    extract_candidates,
    normalize_lemma,
    render_review_images,
    repair_mixed_case_duplicate,
    suffix_base,
)


def token(text: str, left: float, density: float) -> dict:
    return {
        "text": text,
        "left": left,
        "top": 100.0,
        "width": 60.0,
        "height": 24.0,
        "ink_density": density,
    }


class LemmaReviewTests(unittest.TestCase):
    def test_normalizes_stem_boundary_for_game_word(self):
        self.assertEqual(normalize_lemma("amp|el"), "ampel")

    def test_repairs_mixed_case_ocr_duplicate(self):
        self.assertEqual(
            repair_mixed_case_duplicate("-mMässighet"), "-Mässighet"
        )
        self.assertEqual(
            expand_compound(
                "affärs", repair_mixed_case_duplicate("-mMässighet")
            ),
            "affärsmässighet",
        )

    def test_expands_compound_suffix(self):
        self.assertEqual(expand_compound("akademi", "-medlem"), "akademimedlem")

    def test_ocr_l_becomes_compound_boundary_when_order_proves_it(self):
        self.assertEqual(
            infer_compound_series_boundary(
                "affärslangelägenhet", "affär", "anställd"
            ),
            "affärs|angelägenhet",
        )

    def test_compound_series_changes_base_before_following_suffixes(self):
        articles = {
            "pages": [23],
            "articles": [
                {
                    "number": 1,
                    "start_page": 23,
                    "start_column": 1,
                    "start_y": 100.0,
                    "lines": [
                        {
                            "page": 23,
                            "column": 1,
                            "top": 100.0,
                            "bottom": 124.0,
                            "tokens": [
                                token("affär", 100, 0.40),
                                token("(-ä'r)", 200, 0.00),
                                token("-en", 300, 0.10),
                                token("-er", 360, 0.10),
                                token("ss.", 420, 0.10),
                                token("affärslangelägenhet", 500, 0.10),
                            ],
                        },
                        {
                            "page": 23,
                            "column": 1,
                            "top": 140.0,
                            "bottom": 164.0,
                            "tokens": [
                                token("-anställd", 140, 0.10),
                            ],
                        },
                    ],
                }
            ],
        }
        heads = {
            "headwords": [
                {
                    "article_number": 1,
                    "headword": "affär",
                    "stem_headword": "affär",
                }
            ]
        }
        candidates = extract_candidates(articles, heads)
        self.assertEqual(
            [item["lemma"] for item in candidates],
            ["affär", "affärsangelägenhet", "affärsanställd"],
        )

    def test_structured_head_repairs_ocr_headword(self):
        articles = {
            "pages": [23],
            "articles": [
                {
                    "number": 1,
                    "start_page": 23,
                    "start_column": 1,
                    "start_y": 100.0,
                    "lines": [
                        {
                            "page": 23,
                            "column": 1,
                            "top": 100.0,
                            "bottom": 124.0,
                            "tokens": [
                                token("affrikatla", 100, 0.40),
                                token("ss.", 300, 0.10),
                                token("förbindelse", 360, 0.10),
                            ],
                        }
                    ],
                }
            ],
        }
        heads = {
            "headwords": [
                {
                    "article_number": 1,
                    "headword": "affrikat",
                    "stem_headword": "affrikat|a",
                }
            ]
        }
        candidates = extract_candidates(articles, heads)
        self.assertEqual(
            [item["lemma"] for item in candidates],
            ["affrikata"],
        )

    def test_same_printed_row_is_ordered_left_to_right(self):
        right = {
            "source_top": 96.0,
            "source_bottom": 120.0,
            "source_left": 400.0,
        }
        left = {
            "source_top": 100.0,
            "source_bottom": 124.0,
            "source_left": 100.0,
        }
        self.assertEqual(_items_in_reading_order([right, left]), [left, right])
        self.assertEqual(_items_by_printed_row([right, left]), [[left, right]])

    def test_vertical_bar_selects_stem_for_following_suffix(self):
        self.assertEqual(suffix_base("affirm|ation"), "affirm")
        self.assertEqual(
            expand_compound(suffix_base("affirm|ation"), "-era"), "affirmera"
        )

    def test_finds_semibold_word_mid_line_before_its_suffix(self):
        articles = {
            "pages": [23],
            "articles": [
                {
                    "number": 1,
                    "start_page": 23,
                    "start_column": 1,
                    "start_y": 100.0,
                    "lines": [
                        {
                            "page": 23,
                            "column": 1,
                            "top": 100.0,
                            "bottom": 124.0,
                            "tokens": [
                                token("affirmativ", 100, 0.40),
                                token("adj.", 250, 0.10),
                                token("jakande", 320, 0.10),
                                token(",", 410, 0.10),
                            ],
                        },
                        {
                            "page": 23,
                            "column": 1,
                            "top": 140.0,
                            "bottom": 164.0,
                            "tokens": [
                                token("bekräftande", 140, 0.10),
                                token("vanlig", 260, 0.10),
                                token("text", 340, 0.10),
                                token("affirm|ation", 430, 0.10),
                                token("-era", 600, 0.40),
                            ],
                        },
                    ],
                }
            ],
        }
        heads = {
            "headwords": [
                {
                    "article_number": 1,
                    "headword": "affirmativ",
                    "stem_headword": "affirmativ",
                }
            ]
        }
        candidates = extract_candidates(articles, heads)
        self.assertEqual(
            [item["lemma"] for item in candidates],
            ["affirmativ", "affirmation", "affirmera"],
        )

    def test_pronunciation_parenthesis_is_not_a_suffix(self):
        articles = {
            "pages": [23],
            "articles": [
                {
                    "number": 1,
                    "start_page": 23,
                    "start_column": 1,
                    "start_y": 100.0,
                    "lines": [
                        {
                            "page": 23,
                            "column": 1,
                            "top": 100.0,
                            "bottom": 124.0,
                            "tokens": [
                                token("affisch", 100, 0.40),
                                token("s.", 250, 0.10),
                                token("vanlig", 320, 0.10),
                            ],
                        },
                        {
                            "page": 23,
                            "column": 1,
                            "top": 140.0,
                            "bottom": 164.0,
                            "tokens": [
                                token("affischör", 140, 0.40),
                                token("(-ö'r)", 300, 0.00),
                                token("s.", 420, 0.10),
                                token("person", 480, 0.10),
                                token("text", 560, 0.10),
                            ],
                        },
                    ],
                }
            ],
        }
        heads = {
            "headwords": [
                {
                    "article_number": 1,
                    "headword": "affisch",
                    "stem_headword": "affisch",
                }
            ]
        }
        candidates = extract_candidates(articles, heads)
        self.assertEqual(
            [item["lemma"] for item in candidates],
            ["affisch", "affischör"],
        )

    def test_extracts_semibold_word_and_compound_inside_article(self):
        articles = {
            "pages": [23],
            "articles": [
                {
                    "number": 1,
                    "start_page": 23,
                    "start_column": 1,
                    "start_y": 100.0,
                    "lines": [
                        {
                            "page": 23,
                            "column": 1,
                            "top": 100.0,
                            "bottom": 124.0,
                            "tokens": [
                                token("akademi", 100, 0.40),
                                token("s.", 250, 0.10),
                                token("en", 290, 0.10),
                            ],
                        },
                        {
                            "page": 23,
                            "column": 1,
                            "top": 140.0,
                            "bottom": 164.0,
                            "tokens": [
                                token("akademiker", 140, 0.40),
                                token("person", 300, 0.10),
                                token("text", 380, 0.10),
                            ],
                        },
                        {
                            "page": 23,
                            "column": 1,
                            "top": 180.0,
                            "bottom": 204.0,
                            "tokens": [
                                token("–", 130, 0.10),
                                token("-yrke", 170, 0.10),
                                token("-en", 300, 0.40),
                            ],
                        },
                    ],
                }
            ],
        }
        heads = {
            "headwords": [
                {
                    "article_number": 1,
                    "headword": "akademi",
                    "stem_headword": "akademi",
                }
            ]
        }
        candidates = extract_candidates(articles, heads)
        self.assertEqual(
            [item["lemma"] for item in candidates],
            ["akademi", "akademiker", "akademikeryrke"],
        )
        self.assertEqual(candidates[1]["source_page"], 23)
        self.assertEqual(candidates[1]["source_column"], 1)

    def test_renders_two_column_review_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "cache"
            output = root / "review"
            cache.mkdir()
            Image.new("RGB", (400, 300), "white").save(
                cache / "page-0023-deskewed.png"
            )
            items = [
                {
                    "lemma": "akademi",
                    "status": "kandidat",
                    "source_page": 23,
                    "source_column": 1,
                    "source_left": 20.0,
                    "source_right": 90.0,
                    "source_top": 100.0,
                    "source_bottom": 124.0,
                }
            ]
            images = render_review_images(items, [23], cache, output)
            self.assertEqual(len(images), 2)
            self.assertTrue(all(path.exists() for path in images))


if __name__ == "__main__":
    unittest.main()
