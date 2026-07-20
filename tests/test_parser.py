from app.parser import candidate_words, normalize_word, suspicious_word


def test_candidate_words_extracts_only_first_token():
    text = """19
abbé -n -er
fransk katolsk präst
abborr-e -en -ar
A-aktie av serie A
"""
    assert candidate_words(text) == ["abbé", "fransk", "abborr-e", "a-aktie"]


def test_normalize_and_suspicious():
    assert normalize_word("  ^Abbé ") == "abbé"
    assert suspicious_word("tiii") is True
    assert suspicious_word("abborre") is False
