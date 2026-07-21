from app.classifier import WordObservation, train_model
from app.runeberg import (
    _is_printed_page_number,
    instruction_line_keys,
    is_runeberg_instruction_line,
    page_urls,
)


def observation(ink: float, line_left: float, height: float) -> WordObservation:
    return WordObservation(
        text="ord",
        left=10,
        top=10,
        width=30,
        height=12,
        confidence=95.0,
        ink_density=ink,
        line_left=line_left,
        relative_height=height,
    )


def test_page_urls():
    html, image = page_urls(19)
    assert html.endswith("/0019.html")
    assert image.endswith("/0019.3.png")


def test_runeberg_instruction_line_is_removed_as_a_phrase():
    assert is_runeberg_instruction_line(
        "Här nedan syns misstolkade texten från faksimilbilden ovan"
    )
    assert is_runeberg_instruction_line(
        "Below is the raw OCR text from the scanned image"
    )


def test_split_runeberg_overlay_is_removed_as_one_window():
    lines = [
        (("1", "1", "1", "1"), 10, "Här nedan syns"),
        (("1", "1", "1", "2"), 25, "misstolkade texten"),
        (("1", "1", "1", "3"), 40, "från faksimilbilden ovan"),
        (("1", "1", "1", "4"), 100, "abakus -en"),
    ]
    excluded = instruction_line_keys(lines)
    assert excluded == {
        ("1", "1", "1", "1"),
        ("1", "1", "1", "2"),
        ("1", "1", "1", "3"),
    }


def test_single_real_headword_is_not_removed():
    assert not is_runeberg_instruction_line("från")
    assert not is_runeberg_instruction_line("här")


def test_page_number_is_removed_only_near_page_edge():
    assert _is_printed_page_number("19", 5, 12, 1000)
    assert _is_printed_page_number("20.", 930, 12, 1000)
    assert not _is_printed_page_number("19", 400, 12, 1000)
    assert not _is_printed_page_number("nitton", 5, 12, 1000)


def test_model_learns_dark_left_aligned_words():
    samples = []
    for _ in range(30):
        samples.append((observation(0.42, 0.0, 1.08), 1))
        samples.append((observation(0.18, 0.08, 0.96), 0))
    model = train_model(samples)
    assert model.probability(observation(0.45, 0.0, 1.1)) > 0.7
    assert model.probability(observation(0.16, 0.1, 0.9)) < 0.3
