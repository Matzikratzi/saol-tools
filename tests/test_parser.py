from app.parser import candidate_words, normalize_word, suspicious_word


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


def test_normalize_and_suspicious():
    assert normalize_word("  ^Abbé ") == "abbé"
    assert suspicious_word("tiii") is True
    assert suspicious_word("abborre") is False
    assert suspicious_word("efter hand") is True
    assert suspicious_word("-fiske") is True
