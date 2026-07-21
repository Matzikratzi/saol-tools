from app.classifier import WordObservation
from app.main import observations_to_candidates


def obs(text, left, top, width=18, height=20, density=0.20):
    return WordObservation(
        text=text,
        left=left,
        top=top,
        width=width,
        height=height,
        confidence=90.0,
        ink_density=density,
        line_left=0.0,
        relative_height=1.0,
    )


def test_separate_superscript_number_and_one_letter_headword_are_joined():
    observations = [
        obs("¹", 50, 100, width=8, height=10, density=0.06),
        obs("a", 62, 106, width=10, height=16, density=0.08),
        obs("a-n", 92, 105, width=30, density=0.30),
        obs("eller", 130, 105, width=38),
        obs("abakus", 50, 140, width=58, density=0.32),
        obs("-en", 120, 140, width=28),
    ]

    candidates = observations_to_candidates(observations)

    assert candidates[0]["word"] == "a"
    assert candidates[0]["sense_number"] == 1
    assert all(candidate["word"] != "a-n" for candidate in candidates)


def test_first_token_at_column_margin_gets_a_candidate_chance_even_when_thin():
    observations = [
        obs("a", 50, 100, width=10, density=0.03),
        obs("artikeltext", 80, 100, width=80, density=0.18),
        obs("abakus", 50, 135, width=58, density=0.30),
        obs("förklaring", 120, 135, width=80),
    ]

    candidates = observations_to_candidates(observations)

    assert [candidate["word"] for candidate in candidates][:2] == ["a", "abakus"]
    assert candidates[0]["suspicious"] is True


def test_indented_continuation_line_does_not_start_an_article():
    observations = [
        obs("abakus", 50, 100, width=58, density=0.30),
        obs("förklaring", 120, 100, width=80),
        obs("fortsättning", 95, 130, width=90, density=0.32),
        obs("av", 190, 130, width=20),
        obs("abbedissa", 50, 165, width=72, density=0.30),
    ]

    candidates = observations_to_candidates(observations)

    words = [candidate["word"] for candidate in candidates]
    assert words == ["abakus", "abbedissa"]


def test_left_column_is_completed_before_right_column():
    observations = [
        obs("alfa", 40, 100, width=40, density=0.30),
        obs("beta", 40, 300, width=40, density=0.30),
        obs("gamma", 400, 90, width=55, density=0.30),
        obs("delta", 400, 140, width=50, density=0.30),
    ]

    candidates = observations_to_candidates(observations)

    assert [candidate["word"] for candidate in candidates] == [
        "alfa",
        "beta",
        "gamma",
        "delta",
    ]
