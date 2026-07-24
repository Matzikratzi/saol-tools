from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.lemma_review import (
    approve_pages,
    approve_through,
    apply_manual_insertions,
    apply_review_facit,
    _items_by_printed_row,
    _items_in_reading_order,
    display_lemma,
    expand_compound,
    GRAMMAR_MARKERS,
    infer_boundary_at_article_divergence,
    infer_boundary_from_article_family,
    infer_era_boundary_from_verb_grammar,
    infer_boundary_from_previous,
    infer_boundary_from_repeated_suffix,
    infer_compound_series_boundary,
    infer_suffix_boundary_from_series,
    inflection_of_previous,
    inflections_then_part_of_speech,
    merged_alternative_inflection,
    merged_pos_inflection,
    extract_candidates,
    normalize_lemma,
    optional_parenthesis_variants,
    plural_of_previous,
    pronunciation_then_inflection,
    repair_false_boundary_from_runeberg,
    render_review_images,
    recover_runeberg_boundary_series,
    remove_alphabetic_family_outliers,
    repair_initial_i_suffix_from_order,
    repair_compacted_multiword_boundary,
    repair_intrusion_before_boundary,
    repair_mixed_case_duplicate,
    report_html,
    runeberg_short_inflection,
    same_lexical_family,
    suffix_base,
    swedish_sort_key,
    weak_alternative_suffix,
    write_review_bundle,
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

    def test_inflection_dashes_before_part_of_speech_are_not_lemmas(self):
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
                                token("text", 320, 0.10),
                            ],
                        },
                        {
                            "page": 23,
                            "column": 1,
                            "top": 140.0,
                            "bottom": 164.0,
                            "tokens": [
                                token("—er", 140, 0.10),
                                token("s.", 250, 0.10),
                                token("—are", 320, 0.10),
                                token("adj.", 440, 0.10),
                            ],
                        },
                        {
                            "page": 23,
                            "column": 1,
                            "top": 180.0,
                            "bottom": 204.0,
                            "tokens": [
                                token("er", 140, 0.10),
                                token("s.", 250, 0.10),
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
            ["affisch"],
        )

    def test_midline_lemma_followed_by_part_of_speech(self):
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
                                token("agg", 100, 0.58),
                                token("-et", 220, 0.30),
                                token("s.", 300, 0.37),
                                token("ovilja", 360, 0.25),
                                token("avoghet", 470, 0.27),
                                token("aggande", 600, 0.38),
                                token("adj.", 760, 0.29),
                            ],
                        },
                        {
                            "page": 23,
                            "column": 1,
                            "top": 140.0,
                            "bottom": 164.0,
                            "tokens": [
                                token("tärande", 140, 0.28),
                                token("plågande", 280, 0.27),
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
                    "headword": "agg",
                    "stem_headword": "agg",
                }
            ]
        }
        candidates = extract_candidates(articles, heads)
        self.assertEqual(
            [item["lemma"] for item in candidates],
            ["agg", "aggande"],
        )

    def test_aggregation_grammar_sets_base_for_following_compound(self):
        grammar = [
            token("-en", 100, 0.10),
            token("-ers.", 200, 0.10),
        ]
        self.assertTrue(inflections_then_part_of_speech(grammar))
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
                                token("aggregat", 100, 0.46),
                                token("-et", 300, 0.10),
                                token("s.", 380, 0.10),
                                token("enhet", 430, 0.10),
                            ],
                        },
                        {
                            "page": 23,
                            "column": 1,
                            "top": 140.0,
                            "bottom": 164.0,
                            "tokens": [
                                token("aggregation", 140, 0.20),
                                token("-en", 400, 0.10),
                            ],
                        },
                        {
                            "page": 23,
                            "column": 1,
                            "top": 180.0,
                            "bottom": 204.0,
                            "tokens": [
                                token("-ers.", 140, 0.10),
                                token("sammangyttring", 280, 0.10),
                            ],
                        },
                        {
                            "page": 23,
                            "column": 1,
                            "top": 220.0,
                            "bottom": 244.0,
                            "tokens": [
                                token("-s|tillstånd", 140, 0.20),
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
                    "headword": "aggregat",
                    "stem_headword": "aggregat",
                }
            ]
        }
        candidates = extract_candidates(articles, heads)
        self.assertEqual(
            [item["lemma"] for item in candidates],
            ["aggregat", "aggregation", "aggregationstillstånd"],
        )

    def test_agentur_grammar_sets_base_for_following_compound(self):
        self.assertTrue(
            pronunciation_then_inflection(
                [token("(-u'r)", 100, 0.10), token("-en", 200, 0.10)]
            )
        )
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
                                token("agent", 100, 0.40),
                                token("-en", 250, 0.10),
                                token("-ers.", 330, 0.10),
                                token("ombud", 430, 0.10),
                            ],
                        },
                        {
                            "page": 23,
                            "column": 1,
                            "top": 140.0,
                            "bottom": 164.0,
                            "tokens": [
                                token("—", 100, 0.10),
                                token("agentskap", 140, 0.20),
                                token("-et;", 330, 0.10),
                                token("pl.", 410, 0.10),
                                token("=", 470, 0.10),
                                token("s.", 520, 0.10),
                                token("agentur", 570, 0.20),
                                token("(-u'r)", 730, 0.10),
                            ],
                        },
                        {
                            "page": 23,
                            "column": 1,
                            "top": 180.0,
                            "bottom": 204.0,
                            "tokens": [
                                token("-en", 140, 0.10),
                                token("-er", 220, 0.10),
                                token("s.", 300, 0.10),
                                token("verksamhet", 360, 0.10),
                            ],
                        },
                        {
                            "page": 23,
                            "column": 1,
                            "top": 220.0,
                            "bottom": 244.0,
                            "tokens": [
                                token("av", 140, 0.40),
                                token("agent", 220, 0.10),
                                token("o.d.", 330, 0.10),
                                token("-firma", 430, 0.40),
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
                    "headword": "agent",
                    "stem_headword": "agent",
                }
            ]
        }
        candidates = extract_candidates(articles, heads)
        self.assertEqual(
            [item["lemma"] for item in candidates],
            ["agent", "agentskap", "agentur", "agenturfirma"],
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

    def test_agave_plural_plus_noun_marker_is_not_a_lemma(self):
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
                                token("agave", 100, 0.40),
                                token("(-a've)", 230, 0.00),
                                token("-n", 350, 0.10),
                                token("-rs.", 410, 0.10),
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
                    "headword": "agave",
                    "stem_headword": "agave",
                }
            ]
        }
        candidates = extract_candidates(articles, heads)
        self.assertEqual(
            [item["lemma"] for item in candidates],
            ["agave"],
        )

    def test_merged_noun_marker_is_not_a_suffix(self):
        self.assertTrue(merged_pos_inflection("-ers.", "-ers", 0.80))
        self.assertTrue(merged_pos_inflection("-rs.", "-rs", 0.80))
        self.assertTrue(merged_pos_inflection("-ers", "-ers", 0.10))
        self.assertFalse(merged_pos_inflection("-ers", "-ers", 0.80))

    def test_merged_alternative_marker_is_not_a_suffix(self):
        self.assertTrue(merged_alternative_inflection("-etel.", "-etel"))
        self.assertFalse(merged_alternative_inflection("-kredit", "-kredit"))

    def test_long_dash_inflections_identify_a_new_lemma(self):
        self.assertTrue(
            inflections_then_part_of_speech(
                [
                    {"text": "—en"},
                    {"text": "—er"},
                    {"text": "s."},
                ]
            )
        )

    def test_runeberg_restores_false_l_boundary_and_following_tails(self):
        items = [
            {
                "article_number": 1,
                "lemma": "acetyen",
                "stem_lemma": "acetyen",
                "raw": "acety|en",
                "method": "halvfet token",
                "reasons": ["svag halvfetssignal"],
            },
            {
                "article_number": 1,
                "lemma": "acetygass",
                "stem_lemma": "acetygass",
                "raw": "-gaSs",
                "method": "sammansättningssuffix",
                "reasons": [],
            },
            {
                "article_number": 1,
                "lemma": "acetylampa",
                "stem_lemma": "acetylampa",
                "raw": "-lampa",
                "method": "sammansättningssuffix",
                "reasons": [],
            },
            {
                "article_number": 1,
                "lemma": "acetysvetsning",
                "stem_lemma": "acetysvetsning",
                "raw": "-svetsning",
                "method": "sammansättningssuffix",
                "reasons": [],
            },
        ]
        heads = {
            1: {
                "headword": "acetat",
                "runeberg_match_score": 0.95,
                "runeberg_article_lines": [
                    "— acetylen (-e’n) -en ei. -ets. kolväte",
                    "-gas -lampa -svetsning",
                ],
            }
        }
        repair_false_boundary_from_runeberg(items, heads)
        self.assertEqual(
            [item["lemma"] for item in items],
            [
                "acetylen",
                "acetylengas",
                "acetylenlampa",
                "acetylensvetsning",
            ],
        )

    def test_suffix_series_repairs_j_read_for_vertical_boundary(self):
        self.assertEqual(
            infer_suffix_boundary_from_series("-sjanbud", "s"),
            "-s|anbud",
        )

    def test_expands_optional_parenthesized_ending(self):
        self.assertEqual(
            optional_parenthesis_variants("-värld(en)"),
            ["-värld", "-världen"],
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

    def test_pronounced_lemma_and_later_derivative_prove_boundary(self):
        following = [
            token("(-ta'-)", 100, 0.10),
            token("-n", 200, 0.10),
            token("-er", 260, 0.10),
            token("s.", 320, 0.10),
            token("-torisk", 380, 0.40),
        ]
        boundary = infer_boundary_from_repeated_suffix(
            "agitaltor", following
        )
        self.assertEqual(boundary, "agita|tor")
        self.assertEqual(
            expand_compound(suffix_base(boundary), "-torisk"),
            "agitatorisk",
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

    def test_era_boundary_is_recovered_from_verb_grammar(self):
        grammar = [token("-ade", 100, 0.10), token("v.", 200, 0.10)]
        self.assertEqual(
            infer_era_boundary_from_verb_grammar(
                "agglomererl|a", grammar
            ),
            "agglomerer|a",
        )
        self.assertEqual(
            infer_era_boundary_from_verb_grammar(
                "agglutinerla", grammar
            ),
            "agglutiner|a",
        )

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
                                token("agglutination", 100, 0.40),
                                token("-en", 350, 0.10),
                                token("-er", 430, 0.10),
                                token("s.", 510, 0.10),
                                token("agglutinerla", 570, 0.20),
                            ],
                        },
                        {
                            "page": 23,
                            "column": 1,
                            "top": 140.0,
                            "bottom": 164.0,
                            "tokens": [
                                token("-ade", 140, 0.10),
                                token("v.", 230, 0.10),
                                token("hopklumpa", 300, 0.10),
                                token("blodkroppar", 470, 0.10),
                            ],
                        },
                        {
                            "page": 23,
                            "column": 1,
                            "top": 180.0,
                            "bottom": 204.0,
                            "tokens": [
                                token("förändring", 140, 0.10),
                                token("av", 300, 0.10),
                                token("ordstammarna", 370, 0.10),
                                token("-ing", 620, 0.20),
                                token("s.", 720, 0.10),
                                token("—", 780, 0.10),
                            ],
                        },
                        {
                            "page": 23,
                            "column": 1,
                            "top": 220.0,
                            "bottom": 244.0,
                            "tokens": [
                                token("agglutinin", 140, 0.30),
                                token("(-i'n)", 340, 0.10),
                                token("-et", 450, 0.10),
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
                    "headword": "agglutination",
                    "stem_headword": "agglutination",
                }
            ]
        }
        candidates = extract_candidates(articles, heads)
        self.assertEqual(
            [item["lemma"] for item in candidates],
            [
                "agglutination",
                "agglutinera",
                "agglutinering",
                "agglutinin",
            ],
        )

    def test_following_suffix_boundary_repairs_first_series_item(self):
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
                                token("aggression", 100, 0.40),
                                token("-en", 350, 0.10),
                                token("-er", 430, 0.10),
                                token("s.", 510, 0.10),
                                token("-sldrift", 570, 0.20),
                            ],
                        },
                        {
                            "page": 23,
                            "column": 1,
                            "top": 140.0,
                            "bottom": 164.0,
                            "tokens": [
                                token("-s|hämning", 140, 0.20),
                                token("-s|politik", 390, 0.20),
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
                    "headword": "aggression",
                    "stem_headword": "aggression",
                }
            ]
        }
        candidates = extract_candidates(articles, heads)
        self.assertEqual(
            [item["lemma"] for item in candidates],
            [
                "aggression",
                "aggressionsdrift",
                "aggressionshämning",
                "aggressionspolitik",
            ],
        )

    def test_previous_suffix_boundary_repairs_following_suffix(self):
        self.assertEqual(
            infer_suffix_boundary_from_series("-allös", "a"),
            "-a|lös",
        )
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
                                token("agla", 100, 0.40),
                                token("-an", 250, 0.10),
                                token("s.", 330, 0.10),
                            ],
                        },
                        {
                            "page": 23,
                            "column": 1,
                            "top": 140.0,
                            "bottom": 164.0,
                            "tokens": [
                                token("-a|förbud", 140, 0.40),
                                token("-allös", 340, 0.40),
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
                    "headword": "aga",
                    "stem_headword": "ag|a",
                }
            ]
        }
        candidates = extract_candidates(articles, heads)
        self.assertEqual(
            [item["lemma"] for item in candidates],
            ["aga", "agaförbud", "agalös"],
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

    def test_lodstreck_at_article_family_divergence(self):
        self.assertEqual(
            infer_boundary_at_article_divergence(
                "affirmlation",
                "affirmativ",
                [token("-era", 100, 0.10)],
            ),
            "affirm|ation",
        )
        self.assertEqual(
            infer_boundary_at_article_divergence(
                "affirmlation", "affirmativ", []
            ),
            "",
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
                                token("affirmlation", 430, 0.10),
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

    def test_initial_capital_i_suffix_is_repaired_by_order(self):
        following = [token("-man", 500, 0.40)]
        self.assertEqual(
            repair_initial_i_suffix_from_order(
                "-Iokal", "affärs", "affärsliv", following
            ),
            "-lokal",
        )
        self.assertEqual(
            repair_initial_i_suffix_from_order(
                "-Igel", "affärs", "affärsliv", following
            ),
            "-Igel",
        )

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
                                token("-en", 250, 0.10),
                                token("s.", 330, 0.10),
                                token("affärs|angelägenhet", 400, 0.40),
                            ],
                        },
                        {
                            "page": 23,
                            "column": 1,
                            "top": 140.0,
                            "bottom": 164.0,
                            "tokens": [
                                token("-liv", 140, 0.40),
                                token("-Iokal", 280, 0.40),
                                token("-man", 450, 0.40),
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
            [
                "affär",
                "affärsangelägenhet",
                "affärsliv",
                "affärslokal",
                "affärsman",
            ],
        )

    def test_corrupt_short_inflection_uses_aligned_runeberg_grammar(self):
        head = {
            "article_number": 1,
            "headword": "aga",
            "stem_headword": "ag|a",
            "runeberg_line": "^ag|a -an -ors. ä. turkisk titel",
            "runeberg_match_score": 0.96,
        }
        self.assertTrue(runeberg_short_inflection("-OFr", head))

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
                                token("'agla", 100, 0.40),
                                token("-aN", 250, 0.10),
                                token("-OFr", 340, 0.10),
                                token("ä.", 450, 0.10),
                            ],
                        }
                    ],
                }
            ],
        }
        candidates = extract_candidates(
            articles, {"headwords": [head]}
        )
        self.assertEqual(
            [item["lemma"] for item in candidates],
            ["aga"],
        )

    def test_present_form_in_definition_is_not_a_new_lemma(self):
        self.assertTrue(inflection_of_previous("affischera", "affischer"))
        self.assertFalse(inflection_of_previous("affischera", "affischering"))

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
                                token("-en", 250, 0.10),
                                token("-er", 330, 0.10),
                                token("s.", 410, 0.10),
                            ],
                        },
                        {
                            "page": 23,
                            "column": 1,
                            "top": 140.0,
                            "bottom": 164.0,
                            "tokens": [
                                token("—", 100, 0.10),
                                token("affischer|a", 150, 0.20),
                                token("-ade", 360, 0.10),
                            ],
                        },
                        {
                            "page": 23,
                            "column": 1,
                            "top": 180.0,
                            "bottom": 204.0,
                            "tokens": [
                                token("v.", 100, 0.10),
                                token("sätta", 170, 0.10),
                                token("upp", 280, 0.10),
                                token("affischer", 360, 0.20),
                                token("-ing", 560, 0.20),
                            ],
                        },
                        {
                            "page": 23,
                            "column": 1,
                            "top": 220.0,
                            "bottom": 244.0,
                            "tokens": [token("s.", 100, 0.10)],
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
            ["affisch", "affischera", "affischering"],
        )

    def test_runeberg_lodstreck_series_recovers_missing_and_corrupt_words(self):
        head = {
            "article_number": 29,
            "headword": "agitation",
            "runeberg_line": "agitation ( g ) -en -ers. -s|möte -s|tal",
            "runeberg_match_score": 0.86,
        }
        items = [
            {
                "article_number": 29,
                "lemma": "agitation",
                "stem_lemma": "agitation",
                "raw": "agitation",
                "method": "artikelhuvud",
                "source_page": 23,
                "source_column": 2,
                "source_top": 100.0,
                "source_bottom": 124.0,
                "source_left": 100.0,
                "source_right": 300.0,
            },
            {
                "article_number": 29,
                "lemma": "agitationsltal",
                "stem_lemma": "agitationsltal",
                "raw": "-sl|tal",
                "method": "sammansättningssuffix",
                "status": "osäker",
                "reasons": ["svag halvfetssignal"],
            },
            {
                "article_number": 29,
                "lemma": "agitatorisk",
                "stem_lemma": "agitatorisk",
                "raw": "-torisk",
                "method": "sammansättningssuffix",
                "status": "kandidat",
                "reasons": [],
            },
        ]
        recover_runeberg_boundary_series(items, {29: head})
        self.assertEqual(
            [item["lemma"] for item in items],
            [
                "agitation",
                "agitationsmöte",
                "agitationstal",
                "agitatorisk",
            ],
        )
        self.assertEqual(items[1]["method"], "Runebergs lodstrecksserie")
        self.assertEqual(
            items[2]["method"],
            "Runebergkorrigerad lodstrecksserie",
        )

    def test_runeberg_recovery_tolerates_candidate_already_used_as_anchor(self):
        head = {
            "article_number": 1,
            "headword": "bas",
            "runeberg_line": "bas xy x|y",
            "runeberg_match_score": 0.95,
        }
        items = [
            {
                "article_number": 1,
                "lemma": "bas",
                "stem_lemma": "bas",
                "raw": "bas",
                "method": "artikelhuvud",
            },
            {
                "article_number": 1,
                "lemma": "xy",
                "stem_lemma": "xy",
                "raw": "xy",
                "method": "halvfet token",
            },
        ]
        recover_runeberg_boundary_series(items, {1: head})
        self.assertEqual([item["lemma"] for item in items], ["bas", "xy"])

    def test_runeberg_suffix_uses_headword_stem_before_boundary(self):
        head = {
            "article_number": 55,
            "headword": "akacia",
            "stem_headword": "akaci|a",
            "runeberg_stem_headword": "akaci|a",
            "runeberg_line": "akaci|a (aka’-) -an -or s. växt -e|träd",
            "runeberg_match_score": 0.98,
        }
        items = [
            {
                "article_number": 55,
                "lemma": "akacia",
                "stem_lemma": "akaci|a",
                "raw": "akacila",
                "method": "artikelhuvud",
            },
            {
                "article_number": 55,
                "lemma": "akaciaeträd",
                "stem_lemma": "akaciaeträd",
                "raw": "-e|träd",
                "method": "sammansättningssuffix",
                "status": "osäker",
                "reasons": ["svag halvfetssignal"],
            },
        ]
        recover_runeberg_boundary_series(items, {55: head})
        self.assertEqual(
            [item["lemma"] for item in items],
            ["akacia", "akacieträd"],
        )
        self.assertEqual(
            items[1]["method"],
            "Runebergkorrigerad lodstrecksserie",
        )

    def test_full_lodstreck_word_is_placed_after_preceding_compound(self):
        head = {
            "article_number": 31,
            "headword": "agn",
            "runeberg_line": "^agn -et; pi. = s. bete vid fiske -fisk - agn|a",
            "runeberg_match_score": 0.95,
        }
        items = [
            {
                "article_number": 31,
                "lemma": "agn",
                "stem_lemma": "agn",
                "raw": "agn",
                "method": "artikelhuvud",
                "source_page": 24,
                "source_column": 1,
                "source_top": 100.0,
                "source_bottom": 124.0,
                "source_left": 100.0,
                "source_right": 200.0,
            },
            {
                "article_number": 31,
                "lemma": "agnfisk",
                "stem_lemma": "agnfisk",
                "raw": "-fisk",
                "method": "sammansättningssuffix",
                "source_page": 24,
                "source_column": 1,
                "source_top": 100.0,
                "source_bottom": 124.0,
                "source_left": 400.0,
                "source_right": 500.0,
            },
        ]
        recover_runeberg_boundary_series(items, {31: head})
        self.assertEqual(
            [item["lemma"] for item in items],
            ["agn", "agnfisk", "agna"],
        )
        self.assertEqual(items[2]["raw"], "agn|a")
        self.assertEqual(items[2]["method"], "Runebergs lodstrecksserie")
        self.assertGreater(items[2]["source_left"], items[1]["source_right"])
        self.assertEqual(
            [item["lemma"] for item in _items_in_reading_order(items)],
            ["agn", "agnfisk", "agna"],
        )

    def test_manual_facit_insertion_recovers_ocr_omission(self):
        items = [
            {
                "article_number": 26,
                "lemma": "aggressionsdrift",
                "source_page": 23,
                "source_column": 2,
                "source_top": 100.0,
                "source_bottom": 124.0,
                "source_left": 200.0,
                "source_right": 400.0,
            },
            {
                "article_number": 26,
                "lemma": "aggressionshämning",
                "source_page": 23,
                "source_column": 2,
                "source_top": 140.0,
                "source_bottom": 164.0,
                "source_left": 200.0,
                "source_right": 430.0,
            },
        ]
        facit = {
            "manual_insertions": [
                {
                    "after": {
                        "article_number": 26,
                        "lemma": "aggressionsdrift",
                    },
                    "candidate": {
                        "article_number": 26,
                        "lemma": "aggressionshämmad",
                        "stem_lemma": "aggressionshämmad",
                        "raw": "-s|hämmad",
                    },
                }
            ]
        }
        apply_manual_insertions(items, facit)
        self.assertEqual(
            [item["lemma"] for item in items],
            [
                "aggressionsdrift",
                "aggressionshämmad",
                "aggressionshämning",
            ],
        )
        self.assertEqual(items[1]["method"], "facitinsättning")
        self.assertIn("saknades i OCR", items[1]["reasons"][0])
        apply_manual_insertions(items, facit)
        self.assertEqual(len(items), 3)

    def test_word_limited_facit_tracks_exact_approved_prefix(self):
        items = [
            {
                "article_number": 1,
                "lemma": "affisch",
                "source_page": 23,
                "source_column": 1,
                "source_top": 100.0,
                "source_left": 100.0,
            },
            {
                "article_number": 2,
                "lemma": "aga",
                "source_page": 23,
                "source_column": 1,
                "source_top": 200.0,
                "source_left": 100.0,
            },
            {
                "article_number": 3,
                "lemma": "aggressionsdrift",
                "source_page": 23,
                "source_column": 2,
                "source_top": 300.0,
                "source_left": 1600.0,
            },
            {
                "article_number": 3,
                "lemma": "aggressionshämmad",
                "source_page": 23,
                "source_column": 2,
                "source_top": 300.0,
                "source_left": 1900.0,
            },
        ]
        facit = {"version": 1, "pages": {}}
        approve_through(facit, items, "aggressionsdrift")
        self.assertEqual(facit["version"], 2)
        self.assertEqual(
            facit["reviewed_prefix"]["through"]["lemma"],
            "aggressionsdrift",
        )

        missing = apply_review_facit(items, facit)
        self.assertEqual(missing, [])
        self.assertEqual(
            [item["review_state"] for item in items],
            ["approved", "approved", "approved", "unread"],
        )

        changed = [
            items[0].copy(),
            {
                "article_number": 2,
                "lemma": "agaförbud",
                "source_page": 23,
                "source_column": 1,
                "source_top": 240.0,
                "source_left": 100.0,
            },
            items[2].copy(),
            items[3].copy(),
        ]
        missing = apply_review_facit(changed, facit)
        self.assertEqual(
            [item["review_state"] for item in changed],
            ["approved", "facit_new", "approved", "unread"],
        )
        self.assertEqual(
            missing,
            [{"page": 23, "article_number": 2, "lemma": "aga"}],
        )

    def test_word_limited_facit_requires_unique_boundary(self):
        items = [
            {
                "article_number": 1,
                "lemma": "aga",
                "source_page": 23,
            },
            {
                "article_number": 2,
                "lemma": "aga",
                "source_page": 23,
            },
        ]
        with self.assertRaisesRegex(ValueError, "inte entydig"):
            approve_through({"version": 1, "pages": {}}, items, "aga")
        with self.assertRaisesRegex(ValueError, "finns inte"):
            approve_through({"version": 1, "pages": {}}, items, "saknas")

    def test_reviewed_interval_survives_prepended_pages_and_renumbering(self):
        reviewed = [
            {
                "article_number": 1, "lemma": "affirmativ",
                "source_page": 23, "source_column": 1,
                "source_top": 100.0, "source_left": 100.0,
            },
            {
                "article_number": 59, "lemma": "a-kassa",
                "source_page": 24, "source_column": 2,
                "source_top": 1078.0, "source_left": 1413.0,
            },
        ]
        facit = {"version": 1, "pages": {}}
        approve_through(facit, reviewed, "a-kassa")
        rerun = [
            {
                "article_number": 1, "lemma": "förord",
                "source_page": 19, "source_column": 1,
                "source_top": 100.0, "source_left": 100.0,
            },
            {**reviewed[0], "article_number": 101},
            {**reviewed[1], "article_number": 159},
            {
                "article_number": 160, "lemma": "akatalektisk",
                "source_page": 24, "source_column": 2,
                "source_top": 1150.0, "source_left": 1413.0,
            },
        ]
        self.assertEqual(apply_review_facit(rerun, facit), [])
        self.assertEqual(
            [item["review_state"] for item in rerun],
            ["unread", "approved", "approved", "unread"],
        )

    def test_review_facit_marks_matches_and_detects_changes(self):
        items = [
            {
                "article_number": 10,
                "lemma": "afrodisiakum",
                "source_page": 23,
            },
            {
                "article_number": 22,
                "lemma": "aggande",
                "source_page": 23,
            },
            {
                "article_number": 40,
                "lemma": "akademi",
                "source_page": 24,
            },
        ]
        facit = {"version": 1, "pages": {}}
        approve_pages(facit, items, [23])

        missing = apply_review_facit(items, facit)
        self.assertEqual(missing, [])
        self.assertEqual(
            [item["review_state"] for item in items],
            ["approved", "approved", "unread"],
        )
        self.assertEqual(display_lemma(items[0]), "✓ afrodisiakum")

        changed = [
            {
                "article_number": 10,
                "lemma": "afrodisiakum",
                "source_page": 23,
            },
            {
                "article_number": 22,
                "lemma": "agg",
                "source_page": 23,
            },
        ]
        missing = apply_review_facit(changed, facit)
        self.assertEqual(changed[0]["review_state"], "approved")
        self.assertEqual(changed[1]["review_state"], "facit_new")
        self.assertEqual(
            missing,
            [{"page": 23, "article_number": 22, "lemma": "aggande"}],
        )
        self.assertEqual(display_lemma(changed[1]), "⚠ agg")

    def test_el_abbreviation_after_agglutinin_is_not_a_lemma(self):
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
                                token("agglutination", 100, 0.40),
                                token("-en", 350, 0.10),
                                token("s.", 430, 0.10),
                            ],
                        },
                        {
                            "page": 23,
                            "column": 1,
                            "top": 140.0,
                            "bottom": 164.0,
                            "tokens": [
                                token("agglutinin", 140, 0.40),
                                token("(-i'n)", 340, 0.10),
                                token("-et", 450, 0.10),
                                token("el.", 540, 0.40),
                                token("eller", 620, 0.40),
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
                    "headword": "agglutination",
                    "stem_headword": "agglutination",
                }
            ]
        }
        candidates = extract_candidates(articles, heads)
        self.assertEqual(
            [item["lemma"] for item in candidates],
            ["agglutination", "agglutinin"],
        )

    def test_corrupt_short_adjective_t_is_not_a_lemma(self):
        articles = {
            "pages": [24],
            "articles": [
                {
                    "number": 1,
                    "start_page": 24,
                    "start_column": 1,
                    "start_y": 100.0,
                    "lines": [
                        {
                            "page": 24,
                            "column": 1,
                            "top": 100.0,
                            "bottom": 124.0,
                            "tokens": [
                                token("agnat", 100, 0.40),
                                token("-en", 250, 0.10),
                                token("s.", 330, 0.10),
                            ],
                        },
                        {
                            "page": 24,
                            "column": 1,
                            "top": 140.0,
                            "bottom": 164.0,
                            "tokens": [
                                token("-isk", 140, 0.40),
                                token("-Lt", 260, 0.00),
                                token("adj.", 340, 0.10),
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
                    "headword": "agnat",
                    "stem_headword": "agnat",
                    "runeberg_match_score": 0.0,
                }
            ]
        }
        candidates = extract_candidates(articles, heads)
        self.assertEqual(
            [item["lemma"] for item in candidates],
            ["agnat", "agnatisk"],
        )


    def test_repairs_inserted_j_before_agnostiker_boundary(self):
        self.assertEqual(
            repair_intrusion_before_boundary(
                "agnostjliker", "agnosticism"
            ),
            "agnost|iker",
        )
        self.assertEqual(
            repair_intrusion_before_boundary(
                "agnostj|iker", "agnosticism"
            ),
            "agnost|iker",
        )
        self.assertEqual(
            repair_intrusion_before_boundary(
                "affärs|angelägenhet", "affär"
            ),
            "affärs|angelägenhet",
        )

    def test_first_line_alias_and_family_word_are_preserved(self):
        articles = {
            "pages": [24],
            "articles": [
                {
                    "number": 43,
                    "start_page": 24,
                    "start_column": 1,
                    "start_y": 100.0,
                    "lines": [
                        {
                            "page": 24,
                            "column": 1,
                            "top": 100.0,
                            "bottom": 124.0,
                            "tokens": [
                                token("ah", 100, 0.54),
                                token("äv.", 190, 0.32),
                                token("aha", 260, 0.47),
                                token("interj.", 360, 0.27),
                                token("ahaupplevelse", 470, 0.38),
                            ],
                        }
                    ],
                }
            ],
        }
        heads = {
            "headwords": [
                {
                    "article_number": 43,
                    "headword": "ah",
                    "stem_headword": "ah",
                }
            ]
        }
        candidates = extract_candidates(articles, heads)
        self.assertEqual(
            [item["lemma"] for item in candidates],
            ["ah", "aha", "ahaupplevelse"],
        )

    def test_trailing_definition_word_is_bounded_by_next_article(self):
        items = [
            {
                "article_number": 40,
                "lemma": "agremang",
                "raw": "agremang",
                "method": "artikelhuvud",
            },
            {
                "article_number": 40,
                "lemma": "av",
                "raw": "av",
                "method": "halvfet token",
            },
            {
                "article_number": 41,
                "lemma": "agrikultur",
                "raw": "agrikultur",
                "method": "artikelhuvud",
            },
        ]
        heads = {
            40: {"headword": "agremang"},
            41: {"headword": "agrikultur"},
        }
        filtered = remove_alphabetic_family_outliers(items, heads)
        self.assertEqual(
            [item["lemma"] for item in filtered],
            ["agremang", "agrikultur"],
        )

    def test_multiword_next_head_bounds_trailing_definition_word(self):
        items = [
            {
                "article_number": 51,
                "lemma": "ajabaja",
                "raw": "ajabaja",
                "method": "artikelhuvud",
            },
            {
                "article_number": 51,
                "lemma": "av",
                "raw": "av",
                "method": "halvfet token",
            },
            {
                "article_number": 52,
                "lemma": "à jour",
                "raw": "å jour",
                "method": "artikelhuvud",
            },
        ]
        heads = {
            51: {"headword": "ajabaja"},
            52: {"headword": "à jour"},
        }
        filtered = remove_alphabetic_family_outliers(items, heads)
        self.assertEqual(
            [item["lemma"] for item in filtered],
            ["ajabaja", "à jour"],
        )

    def test_compacted_multiword_head_starts_compound_series(self):
        self.assertEqual(
            repair_compacted_multiword_boundary(
                "åjourl|föra,", "à jour"
            ),
            "ajour|föra",
        )
        articles = {
            "pages": [24],
            "articles": [
                {
                    "number": 52,
                    "start_page": 24,
                    "start_column": 1,
                    "start_y": 100.0,
                    "lines": [
                        {
                            "page": 24,
                            "column": 1,
                            "top": 100.0,
                            "bottom": 124.0,
                            "tokens": [
                                token("å", 100, 0.54),
                                token("jour", 160, 0.47),
                                token("oböjl.", 280, 0.28),
                                token("adj.;", 390, 0.27),
                            ],
                        },
                        {
                            "page": 24,
                            "column": 1,
                            "top": 130.0,
                            "bottom": 154.0,
                            "tokens": [
                                token("åjourl|föra,", 100, 0.30),
                                token("-föring", 350, 0.35),
                                token("s.,", 500, 0.28),
                                token("-hålla,", 580, 0.35),
                                token("-hållning;", 730, 0.32),
                            ],
                        },
                    ],
                }
            ],
        }
        heads = {
            "headwords": [
                {
                    "article_number": 52,
                    "headword": "à jour",
                    "raw_headword": "å jour",
                    "stem_headword": "å jour",
                }
            ]
        }
        candidates = extract_candidates(articles, heads)
        self.assertEqual(
            [item["lemma"] for item in candidates],
            [
                "à jour",
                "ajourföra",
                "ajourföring",
                "ajourhålla",
                "ajourhållning",
            ],
        )

    def test_later_inflection_does_not_turn_definition_into_compound_base(self):
        articles = {
            "pages": [24],
            "articles": [
                {
                    "number": 56,
                    "start_page": 24,
                    "start_column": 1,
                    "start_y": 100.0,
                    "lines": [
                        {
                            "page": 24,
                            "column": 1,
                            "top": 100.0,
                            "bottom": 124.0,
                            "tokens": [
                                token("akademi", 100, 0.50),
                                token("-en", 300, 0.25),
                                token("s.", 400, 0.30),
                            ],
                        },
                        {
                            "page": 24,
                            "column": 2,
                            "top": 130.0,
                            "bottom": 154.0,
                            "tokens": [
                                token("läroanstalt", 100, 0.36),
                                token("m.m.", 390, 0.34),
                                token("-hemman", 500, 0.41),
                            ],
                        },
                        {
                            "page": 24,
                            "column": 2,
                            "top": 160.0,
                            "bottom": 184.0,
                            "tokens": [
                                token("-ledamot", 100, 0.44),
                                token("-medlem", 320, 0.43),
                                token("-räntmästare", 520, 0.46),
                            ],
                        },
                        {
                            "page": 24,
                            "column": 2,
                            "top": 190.0,
                            "bottom": 214.0,
                            "tokens": [
                                token("-sekreterare", 100, 0.44),
                                token("—", 430, 0.70),
                                token("akademiker", 500, 0.48),
                            ],
                        },
                        {
                            "page": 24,
                            "column": 2,
                            "top": 220.0,
                            "bottom": 244.0,
                            "tokens": [
                                token("akademisk", 100, 0.46),
                                token("-t", 400, 0.25),
                                token("adj.", 470, 0.28),
                            ],
                        },
                    ],
                }
            ],
        }
        heads = {
            "headwords": [
                {
                    "article_number": 56,
                    "headword": "akademi",
                    "stem_headword": "akademi",
                }
            ]
        }
        candidates = extract_candidates(articles, heads)
        self.assertEqual(
            [item["lemma"] for item in candidates],
            [
                "akademi",
                "akademihemman",
                "akademiledamot",
                "akademimedlem",
                "akademiräntmästare",
                "akademisekreterare",
                "akademiker",
                "akademisk",
            ],
        )

    def test_unbold_alternative_suffix_in_definition_is_rejected(self):
        self.assertTrue(weak_alternative_suffix("el.", 0.0))
        self.assertTrue(weak_alternative_suffix("eller", 0.44))
        self.assertFalse(weak_alternative_suffix("el.", 0.45))
        articles = {
            "pages": [23],
            "articles": [
                {
                    "number": 8,
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
                                token("afrikan", 100, 0.50),
                                token("-en", 260, 0.10),
                                token("s.", 340, 0.10),
                            ],
                        },
                        {
                            "page": 23,
                            "column": 1,
                            "top": 140.0,
                            "bottom": 164.0,
                            "tokens": [
                                token("afrik|anist", 140, 0.30),
                                token("(-ist')", 360, 0.10),
                                token("-en", 480, 0.10),
                                token("s.", 560, 0.10),
                            ],
                        },
                        {
                            "page": 23,
                            "column": 1,
                            "top": 180.0,
                            "bottom": 204.0,
                            "tokens": [
                                token("afrikaforskare", 140, 0.20),
                                token("el.", 430, 0.20),
                                token("-kännare", 500, 0.20),
                                token("-ansk", 680, 0.50),
                            ],
                        },
                    ],
                }
            ],
        }
        heads = {
            "headwords": [
                {
                    "article_number": 8,
                    "headword": "afrikan",
                    "stem_headword": "afrikan",
                }
            ]
        }
        candidates = extract_candidates(articles, heads)
        self.assertEqual(
            [item["lemma"] for item in candidates],
            ["afrikan", "afrikanist", "afrikansk"],
        )

    def test_compound_before_new_family_base_is_preserved(self):
        self.assertTrue(same_lexical_family("agn", "agnfisk"))
        self.assertTrue(same_lexical_family("agn", "agna"))
        items = [
            {
                "article_number": 31,
                "lemma": "agn",
                "raw": "agn",
                "method": "artikelhuvud",
            },
            {
                "article_number": 31,
                "lemma": "agnfisk",
                "raw": "-fisk",
                "method": "sammansättningssuffix",
            },
            {
                "article_number": 31,
                "lemma": "agna",
                "raw": "agn|a",
                "method": "Runebergs lodstrecksserie",
            },
        ]
        filtered = remove_alphabetic_family_outliers(
            items, {31: {"headword": "agn"}}
        )
        self.assertEqual(
            [item["lemma"] for item in filtered],
            ["agn", "agnfisk", "agna"],
        )

    def test_definition_words_are_rejected_by_alphabetic_interval(self):
        self.assertNotIn("som", GRAMMAR_MARKERS)
        self.assertNotIn("att", GRAMMAR_MARKERS)
        self.assertNotIn("om", GRAMMAR_MARKERS)
        self.assertLess(
            swedish_sort_key("agnosticism"),
            swedish_sort_key("agnostiker"),
        )
        items = [
            {
                "article_number": 34,
                "lemma": "agnosticism",
                "raw": "agnosticism",
                "method": "artikelhuvud",
            },
            {
                "article_number": 34,
                "lemma": "som",
                "raw": "som",
                "method": "halvfet token",
            },
            {
                "article_number": 34,
                "lemma": "att",
                "raw": "att",
                "method": "halvfet token",
            },
            {
                "article_number": 34,
                "lemma": "om",
                "raw": "om",
                "method": "halvfet token",
            },
            {
                "article_number": 34,
                "lemma": "agnostiker",
                "raw": "agnost|iker",
                "method": "halvfet token",
            },
        ]
        heads = {34: {"headword": "agnosticism"}}
        filtered = remove_alphabetic_family_outliers(items, heads)
        self.assertEqual(
            [item["lemma"] for item in filtered],
            ["agnosticism", "agnostiker"],
        )

    def test_agnosticism_definition_words_are_not_lemmas(self):
        articles = {
            "pages": [24],
            "articles": [
                {
                    "number": 34,
                    "start_page": 24,
                    "start_column": 1,
                    "start_y": 100.0,
                    "lines": [
                        {
                            "page": 24,
                            "column": 1,
                            "top": 100.0,
                            "bottom": 124.0,
                            "tokens": [
                                token("agnosticism", 100, 0.40),
                                token("-en", 350, 0.10),
                                token("s.", 430, 0.10),
                            ],
                        },
                        {
                            "page": 24,
                            "column": 1,
                            "top": 140.0,
                            "bottom": 164.0,
                            "tokens": [
                                token("som", 140, 0.50),
                                token("att", 240, 0.50),
                                token("om", 340, 0.50),
                            ],
                        },
                        {
                            "page": 24,
                            "column": 1,
                            "top": 180.0,
                            "bottom": 204.0,
                            "tokens": [
                                token("agnostjliker", 140, 0.30),
                                token("(-gnåst'-)", 400, 0.10),
                                token("-n", 550, 0.10),
                                token("-iker", 640, 0.20),
                                token("s.", 760, 0.10),
                                token("-isk", 820, 0.45),
                            ],
                        },
                    ],
                }
            ],
        }
        heads = {
            "headwords": [
                {
                    "article_number": 34,
                    "headword": "agnosticism",
                    "stem_headword": "agnosticism",
                    "runeberg_match_score": 0.0,
                }
            ]
        }
        candidates = extract_candidates(articles, heads)
        self.assertEqual(
            [item["lemma"] for item in candidates],
            ["agnosticism", "agnostiker", "agnostisk"],
        )

    def test_report_links_to_first_unread_column(self):
        items = [
            {
                "article_number": 1,
                "page": 24,
                "column": 1,
                "lemma": "agnosticism",
                "method": "artikelhuvud",
                "bold_score": 1.0,
                "status": "kandidat",
                "reasons": [],
                "raw": "agnosticism",
                "source_page": 24,
                "source_column": 1,
                "review_state": "approved",
            },
            {
                "article_number": 1,
                "page": 24,
                "column": 1,
                "lemma": "agnostiker",
                "method": "halvfet token",
                "bold_score": 0.3,
                "status": "osäker",
                "reasons": ["svag halvfetssignal"],
                "raw": "agnost|iker",
                "source_page": 24,
                "source_column": 1,
                "source_top": 100.0,
                "review_state": "unread",
            },
        ]
        output = report_html(
            items,
            [Path("lemma-review-pages/page-0024-column-1.png")],
        )
        self.assertIn(
            'href="#first-review"', output
        )
        self.assertIn(
            'id="review-page-24-column-1"', output
        )
        self.assertIn('id="first-review"', output)

    def test_packages_three_review_json_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for name in (
                "article-text-review.json",
                "headword-review.json",
                "lemma-review.json",
            ):
                path = root / name
                path.write_text('{"ok": true}', encoding="utf-8")
                paths.append(path)
            bundle = write_review_bundle(root / "reviews.zip", paths)
            self.assertTrue(bundle.exists())
            import zipfile
            with zipfile.ZipFile(bundle) as archive:
                self.assertEqual(
                    archive.namelist(),
                    [path.name for path in paths],
                )
                self.assertEqual(
                    archive.read("lemma-review.json"),
                    b'{"ok": true}',
                )

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
