from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.lemma_review import (
    expand_compound,
    extract_candidates,
    normalize_lemma,
    render_review_images,
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

    def test_expands_compound_suffix(self):
        self.assertEqual(expand_compound("akademi", "-medlem"), "akademimedlem")

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
