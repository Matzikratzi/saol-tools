from app.classifier import WordObservation, train_model
from app.parser import normalize_word
from app.runeberg import (
    _is_printed_page_number,
    _stem_marked_tokens,
    instruction_line_keys,
    is_runeberg_instruction_line,
    page_urls,
    reconcile_stem_marked_observations,
)


def observation(
    ink: float,
    line_left: float,
    height: float,
    text: str = "ord",
) -> WordObservation:
    return WordObservation(
        text=text,
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


def test_runeberg_stem_marked_token_is_extracted_from_html():
    html = "<html><body>OCR: abbrevi|ation och abborr|e.</body></html>"
    assert _stem_marked_tokens(html) == ["abbrevi|ation", "abborr|e"]


def test_runeberg_ocr_corrects_tesseract_l_without_losing_geometry():
    original = observation(0.42, 0.0, 1.08, text="abbrevilation")
    corrected = reconcile_stem_marked_observations(
        [original], ["abbrevi|ation"]
    )

    assert corrected[0].text == "abbrevi|ation"
    assert normalize_word(corrected[0].text) == "abbreviation"
    assert corrected[0].left == original.left
    assert corrected[0].top == original.top
    assert corrected[0].width == original.width
    assert corrected[0].height == original.height


def test_model_learns_dark_left_aligned_words():
    samples = []
    for _ in range(30):
        samples.append((observation(0.42, 0.0, 1.08), 1))
        samples.append((observation(0.18, 0.08, 0.96), 0))
    model = train_model(samples)
    assert model.probability(observation(0.45, 0.0, 1.1)) > 0.7
    assert model.probability(observation(0.16, 0.1, 0.9)) < 0.3
