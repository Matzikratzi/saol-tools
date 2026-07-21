from app.parser import (
    candidate_words,
    normalize_forms,
    normalize_word,
    split_headword_marker,
    suspicious_word,
)


def test_candidate_words_preserves_complete_bold_groups():
    text = """19
abbé
a-aktie
efter hand
-fiske
"""
    assert candidate_words(text) == ["abbé", "a-aktie", "efter hand", "-fiske"]


def test_candidate_words_ignores_runeberg_metadata():
    text = """Below is the raw OCR text from the above scanned image.
Do you see an error? Proofread the page now!
Här nedan syns maskintolkade texten från faksimilbilden ovan.
Ser du något fel? Korrekturläs sidan nu!
This page has never been proofread. / Denna sida har aldrig korrekturlästs.
19
abakus
"""
    assert candidate_words(text) == ["abakus"]


def test_homonym_number_is_metadata_not_headword_text():
    assert split_headword_marker("¹a") == (1, "a")
    assert split_headword_marker("²a") == (2, "a")
    assert split_headword_marker("³a") == (3, "a")
    assert split_headword_marker("1a") == (1, "a")
    assert split_headword_marker("12abc") == (12, "abc")
    assert split_headword_marker("a1") == (None, "a1")


def test_equal_headwords_with_different_numbers_remain_separate_articles():
    assert candidate_words("¹a\n²a\n³a") == ["a", "a", "a"]


def test_game_forms_are_split_normalized_and_deduplicated():
    assert normalize_forms(["Katten, katter", "katterna; katten"]) == [
        "katten",
        "katter",
        "katterna",
    ]


def test_normalize_and_suspicious():
    assert normalize_word("  ^Abbé ") == "abbé"
    assert suspicious_word("tiii") is True
    assert suspicious_word("abborre") is False
    assert suspicious_word("efter hand") is True
    assert suspicious_word("-fiske") is True
