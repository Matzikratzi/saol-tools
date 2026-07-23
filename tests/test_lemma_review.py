from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.lemma_review import (
    _items_by_printed_row,
    _items_in_reading_order,
    display_lemma,
    expand_compound,
    infer_boundary_from_article_family,
    infer_boundary_from_previous,
    infer_boundary_from_repeated_suffix,
    infer_compound_series_boundary,
    inflection_of_previous,
    merged_pos_inflection,
    extract_candidates,
    normalize_lemma,
    optional_parenthesis_variants,
    plural_of_previous,
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
    def test_displays_and_preserves_homonym_number(self):
        item = {
            "lemma": "afro",
            "method": "artikelhuvud",
            "homonym": 2,
        }
        self.assertEqual(display_lemma(item), "[H2] afro")
        item["method"] = "sammansättningssuffix"
        self.assertEqual(display_lemma(item), "afro")

    def test_normalizes_stem_boundary_for_game_word(self):
        self.assertEqual(normalize_lemma("amp|el"), "ampel")

    def test_plural_of_previous_a_noun_is_not_a_lemma(self):
        self.assertTrue(plural_of_previous("afrikanska", "afrikanskor"))
        self.assertFalse(plural_of_previous("afrikansk", "afrikanskor"))

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
                                token("afrik", 100, 0.40),
                                token("s.", 250, 0.10),
                                token("ord", 320, 0.10),
                                token("text", 390, 0.10),
                            ],
                        },
                        {
                            "page": 23,
                            "column": 1,
                            "top": 140.0,
                            "bottom": 164.0,
                            "tokens": [
                                token("-anska", 140, 0.40),
                                token("-n", 300, 0.10),
                                token("-anskor", 360, 0.10),
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
                    "headword": "afrik",
                    "stem_headword": "afrik",
                }
            ]
        }
        candidates = extract_candidates(articles, heads)
        self.assertEqual(
            [item["lemma"] for item in candidates],
            ["afrik", "afrikanska"],
        )

    def test_full_token_definite_form_is_not_a_lemma(self):
        self.assertTrue(
            inflection_of_previous("afrodisiakum", "afrodisiakumet")
        )
        self.assertFalse(
            inflection_of_previous("afrodisiakum", "afrodisiakum")
        )

    def test_weak_line_start_definition_is_not_a_lemma(self):
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
                                token("afrodisiak|um", 100, 0.40),
                                token("-umet", 300, 0.10),
                                token("s.", 370, 0.10),
                                token("medel", 430, 0.10),
                            ],
                        },
                        {
                            "page": 23,
                            "column": 1,
                            "top": 140.0,
                            "bottom": 164.0,
                            "tokens": [
                                token("mest", 140, 0.229),
                                token("förklarande", 250, 0.10),
                                token("text", 430, 0.10),
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
                    "headword": "afrodisiakum",
                    "stem_headword": "afrodisiak|um",
                }
            ]
        }
        candidates = extract_candidates(articles, heads)
        self.assertEqual(
            [item["lemma"] for item in candidates],
            ["afrodisiakum"],
        )

    def test_stem_suffix_inflection_is_not_a_lemma(self):
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
                                token("afrodisiak|um", 100, 0.40),
                                token("-umet", 300, 0.10),
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
                    "headword": "afrodisiakum",
                    "stem_headword": "afrodisiak|um",
                }
            ]
        }
        candidates = extract_candidates(articles, heads)
        self.assertEqual(
            [item["lemma"] for item in candidates],
            ["afrodisiakum"],
        )

    def test_merged_noun_marker_is_not_a_suffix(self):
        self.assertTrue(merged_pos_inflection("-ers.", "-ers", 0.80))
        self.assertTrue(merged_pos_inflection("-ers", "-ers", 0.10))
        self.assertFalse(merged_pos_inflection("-ers", "-ers", 0.80))

    def test_expands_optional_parenthesized_ending(self):
        self.assertEqual(
            optional_parenthesis_variants("-värld(en)"),
            ["-världen"],
        )

    def test_full_word_followed_by_same_suffix_is_not_duplicated(self):
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
                                token("afghan", 100, 0.40),
                                token("s.", 250, 0.10),
                                token("inv.", 310, 0.40),
                                token("i", 370, 0.40),
                                token("Afghanistan", 400, 0.10),
                                token("vanlig", 520, 0.10),
                                token("förklarande", 600, 0.10),
                                token("text", 720, 0.10),
                                token("om", 790, 0.10),
                                token("person", 850, 0.10),
                            ],
                        },
                        {
                            "page": 23,
                            "column": 1,
                            "top": 140.0,
                            "bottom": 164.0,
                            "tokens": [
                                token("afghanhund", 140, 0.40),
                                token("-hund", 350, 0.40),
                                token("—", 470, 0.10),
                                token("afghansk", 520, 0.10),
                            ],
                        },
                        {
                            "page": 23,
                            "column": 1,
                            "top": 180.0,
                            "bottom": 204.0,
                            "tokens": [
                                token("afghanskla", 140, 0.10),
                                token("-an", 350, 0.10),
                                token("-or", 430, 0.10),
                                token("s.", 500, 0.10),
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
                    "headword": "afghan",
                    "stem_headword": "afghan",
                }
            ]
        }
        candidates = extract_candidates(articles, heads)
        self.assertEqual(
            [item["lemma"] for item in candidates],
            ["afghan", "afghanhund", "afghansk", "afghanska"],
        )

    def test_article_family_proves_l_is_boundary(self):
        self.assertEqual(
            infer_boundary_from_article_family("afrikalresa", "afrikan"),
            "afrika|resa",
        )

    def test_previous_family_word_proves_l_is_boundary(self):
        self.assertEqual(
            infer_boundary_from_previous("afghanskla", "afghansk"),
            "afghansk|a",
        )

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

    def test_repeated_suffix_proves_l_is_a_boundary(self):
        following = [
            token("-n", 100, 0.10),
            token("-iker", 200, 0.10),
            token("-isk", 300, 0.10),
        ]
        self.assertEqual(
            infer_boundary_from_repeated_suffix(
                "aforistliker", following
            ),
            "aforist|iker",
        )

    def test_aforism_family_ignores_definition_and_changes_base(self):
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
                                token("aforism", 100, 0.40),
                                token("s.", 250, 0.10),
                                token("tänkespråk", 320, 0.10),
                                token("vanlig", 450, 0.10),
                            ],
                        },
                        {
                            "page": 23,
                            "column": 1,
                            "top": 140.0,
                            "bottom": 164.0,
                            "tokens": [
                                token("-samling", 140, 0.40),
                                token("—", 300, 0.10),
                                token("aforistik", 350, 0.40),
                            ],
                        },
                        {
                            "page": 23,
                            "column": 1,
                            "top": 180.0,
                            "bottom": 204.0,
                            "tokens": [
                                token("litteratur", 140, 0.10),
                                token("i", 280, 0.40),
                                token("form", 320, 0.10),
                                token("av", 390, 0.40),
                                token("aforismer", 450, 0.10),
                            ],
                        },
                        {
                            "page": 23,
                            "column": 1,
                            "top": 220.0,
                            "bottom": 244.0,
                            "tokens": [
                                token("aforistliker", 140, 0.40),
                                token("(-ist'-)", 360, 0.00),
                                token("-n", 480, 0.10),
                                token("pl.", 530, 0.10),
                                token("-iker", 590, 0.10),
                                token("s.", 680, 0.10),
                                token("-isk", 730, 0.40),
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
                    "headword": "aforism",
                    "stem_headword": "aforism",
                }
            ]
        }
        candidates = extract_candidates(articles, heads)
        self.assertEqual(
            [item["lemma"] for item in candidates],
            [
                "aforism",
                "aforismsamling",
                "aforistik",
                "aforistiker",
                "aforistisk",
            ],
        )

    def test_definition_word_does_not_replace_stem_before_later_suffix(self):
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
                                token("afrikan", 100, 0.40),
                                token("-en", 250, 0.10),
                                token("-er", 310, 0.10),
                                token("s.", 370, 0.10),
                                token("person", 430, 0.10),
                                token("text", 510, 0.10),
                            ],
                        },
                        {
                            "page": 23,
                            "column": 1,
                            "top": 140.0,
                            "bottom": 164.0,
                            "tokens": [
                                token("afrikaniser|a", 140, 0.40),
                                token("-ade", 360, 0.10),
                                token("v.", 430, 0.10),
                                token("göra", 480, 0.10),
                                token("mera", 550, 0.40),
                                token("afrikansk", 640, 0.10),
                            ],
                        },
                        {
                            "page": 23,
                            "column": 1,
                            "top": 180.0,
                            "bottom": 204.0,
                            "tokens": [
                                token("vanlig", 140, 0.10),
                                token("förklarande", 250, 0.10),
                                token("text", 430, 0.10),
                                token("-ing", 520, 0.40),
                            ],
                        },
                        {
                            "page": 23,
                            "column": 1,
                            "top": 220.0,
                            "bottom": 244.0,
                            "tokens": [
                                token("afrikalresa", 140, 0.40),
                            ],
                        },
                        {
                            "page": 23,
                            "column": 1,
                            "top": 260.0,
                            "bottom": 284.0,
                            "tokens": [
                                token("afro|amerikan", 140, 0.40),
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
                    "headword": "afrikan",
                    "stem_headword": "afrikan",
                }
            ]
        }
        candidates = extract_candidates(articles, heads)
        self.assertEqual(
            [item["lemma"] for item in candidates],
            [
                "afrikan",
                "afrikanisera",
                "afrikanisering",
                "afrikaresa",
                "afroamerikan",
            ],
        )

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
