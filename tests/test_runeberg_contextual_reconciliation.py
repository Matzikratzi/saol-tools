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
    assert corrected[1].left == observations[1].left
    assert corrected[1].top == observations[1].top
    assert corrected[1].width == observations[1].width
    assert corrected[1].height == observations[1].height


def test_contextual_repair_does_not_replace_unrelated_readable_word():
    observations = [
        observation("ablativ", 100),
        observation("abborre", 120),
        observation("abolition", 140),
    ]

    corrected = reconcile_contextual_observations(
        observations,
        ["ablativ", "abnorm", "abolition"],
    )

    assert corrected[1].text == "abborre"
