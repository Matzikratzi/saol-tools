from app.parser import candidate_words, normalize_word, suspicious_word


def test_candidate_words_extracts_only_first_token():
    text = """19
abbé -n -er
fransk katolsk präst
abborr-e -en -ar
A-aktie av serie A
"""
    assert candidate_words(text) == ["abbé", "fransk", "abborr-e", "a-aktie"]


def test_candidate_words_ignores_runeberg_metadata():
    text = """Below is the raw OCR text from the above scanned image.
Do you see an error? Proofread the page now!
Här nedan syns maskintolkade texten från faksimilbilden ovan.
Ser du något fel? Korrekturläs sidan nu!
This page has never been proofread. / Denna sida har aldrig korrekturlästs.
19
abakus -en -er
"""
    assert candidate_words(text) == ["abakus"]


def test_normalize_and_suspicious():
    assert normalize_word("  ^Abbé ") == "abbé"
    assert suspicious_word("tiii") is True
    assert suspicious_word("abborre") is False
