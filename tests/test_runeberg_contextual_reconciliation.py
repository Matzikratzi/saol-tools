from app.classifier import WordObservation
from app.runeberg import reconcile_contextual_observations


def observation(text: str, top: int, left: int = 600) -> WordObservation:
    return WordObservation(
        text=text,
        left=left,
        top=top,
        width=70,
        height=20,
        confidence=80.0,
        ink_density=0.2,
        line_left=0.0,
        relative_height=1.0,
    )


def test_runeberg_repairs_isolated_unreadable_word_by_context():
    observations = [
        observation("ablativ", 100),
        observation("—", 120),
        observation("abolition", 140),
    ]

    corrected = reconcile_contextual_observations(
        observations,
        ["ablativ", "abnorm", "abolition"],
    )

    assert [item.text for item in corrected] == ["ablativ", "abnorm", "abolition"]
    assert corrected[1].ocr_tesseract == "—"
    assert corrected[1].ocr_runeberg == "abnorm"
    assert corrected[1].ocr_conflict is True
    assert corrected[1].left == observations[1].left
    assert corrected[1].top == observations[1].top
    assert corrected[1].width == observations[1].width
    assert corrected[1].height == observations[1].height


def test_contextual_repair_preserves_unrelated_readable_word_as_conflict():
    observations = [
        observation("ablativ", 100),
        observation("abborre", 120),
        observation("abolition", 140),
    ]

    corrected = reconcile_contextual_observations(
        observations,
        ["ablativ", "abnorm", "abolition"],
    )

    conflict = corrected[1]
    assert conflict.text == "abnorm"
    assert conflict.ocr_runeberg == "abnorm"
    assert conflict.ocr_tesseract == "abborre"
    assert conflict.ocr_conflict is True


def test_contextual_repair_handles_equal_sized_multi_token_replacement_block():
    observations = [
        observation("ablativ", 100),
        observation("—", 120),
        observation("felord", 140),
        observation("abolition", 160),
    ]

    corrected = reconcile_contextual_observations(
        observations,
        ["ablativ", "abnorm", "rättord", "abolition"],
    )

    assert corrected[1].text == "abnorm"
    assert corrected[1].ocr_tesseract == "—"
    assert corrected[1].ocr_runeberg == "abnorm"
    assert corrected[1].ocr_conflict is True
    assert corrected[2].text == "felord"
    assert corrected[2].ocr_runeberg == ""
