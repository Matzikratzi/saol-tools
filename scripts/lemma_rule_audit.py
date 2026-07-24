from __future__ import annotations

"""Measure lemma-rule use and disable rules one at a time."""

import argparse
import json
from pathlib import Path

import scripts.lemma_review as lemma_review


ROOT = Path(__file__).resolve().parents[1]


NEUTRAL_RULES = {
    "repair_initial_i_suffix_from_order": lambda value, *args: value,
    "repair_mixed_case_duplicate": lambda value: value,
    "repair_intrusion_before_boundary": lambda value, *args: value,
    "repair_compacted_multiword_boundary": lambda value, *args: value,
    "infer_era_boundary_from_verb_grammar": lambda *args: "",
    "infer_boundary_from_previous": lambda *args: "",
    "infer_suffix_boundary_from_series": lambda *args: "",
    "infer_boundary_at_article_divergence": lambda *args: "",
    "infer_boundary_from_article_family": lambda *args: "",
    "infer_boundary_from_repeated_suffix": lambda *args: "",
    "infer_compound_series_boundary": lambda *args: "",
    "recover_runeberg_boundary_series": lambda items, heads: items,
    "remove_alphabetic_family_outliers": lambda items, heads: items,
    "weak_alternative_suffix": lambda *args: False,
    "runeberg_short_inflection": lambda *args: False,
    "plural_of_previous": lambda *args: False,
    "present_form_of_previous": lambda *args: False,
}


def signatures(items: list[dict]) -> list[tuple[int, str]]:
    return [
        (int(item["article_number"]), item["lemma"])
        for item in items
    ]


def extract(articles: dict, heads: dict, facit: dict) -> list[dict]:
    items = lemma_review.extract_candidates(articles, heads)
    lemma_review.apply_manual_insertions(items, facit)
    return items


def facit_deviations(
    items: list[dict], facit: dict
) -> tuple[int, int]:
    """Return unexpected and missing candidates inside reviewed material."""
    missing = lemma_review.apply_review_facit(items, facit)
    unexpected = sum(
        item.get("review_state") == "facit_new" for item in items
    )
    return unexpected, len(missing)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--articles",
        type=Path,
        default=Path("article-text-review.json"),
    )
    parser.add_argument(
        "--headwords",
        type=Path,
        default=Path("headword-review.json"),
    )
    parser.add_argument(
        "--facit",
        type=Path,
        default=ROOT / "data" / "lemma_review_facit.json",
    )
    parser.add_argument(
        "--headword-corrections",
        type=Path,
        default=ROOT / "data" / "headword_corrections.json",
    )
    args = parser.parse_args()

    articles = json.loads(args.articles.read_text(encoding="utf-8"))
    heads = json.loads(args.headwords.read_text(encoding="utf-8"))
    facit = json.loads(args.facit.read_text(encoding="utf-8"))

    baseline_items = extract(articles, heads, facit)
    baseline = signatures(baseline_items)
    baseline_set = set(baseline)
    baseline_facit = facit_deviations(baseline_items, facit)
    baseline_hits = lemma_review.rule_stats()

    print(f"Grundresultat: {len(baseline)} kandidater")
    print(
        "Facitavvikelser i grundresultatet: "
        f"+{baseline_facit[0]} −{baseline_facit[1]}"
    )
    print("\nREGELTRÄFFAR")
    for name, count in baseline_hits.items():
        print(f"{count:4d}  {name}")

    print("\nABLATION – EN REGEL AVSTÄNGD I TAGET")
    for name, neutral in NEUTRAL_RULES.items():
        original = getattr(lemma_review, name)
        setattr(lemma_review, name, neutral)
        try:
            changed_items = extract(articles, heads, facit)
        finally:
            setattr(lemma_review, name, original)
        changed = signatures(changed_items)
        changed_set = set(changed)
        added = len(changed_set - baseline_set)
        missing = len(baseline_set - changed_set)
        facit_unexpected, facit_missing = facit_deviations(
            changed_items, facit
        )
        state = (
            "OFÖRÄNDRAT"
            if changed == baseline
            else f"+{added} −{missing}, totalt {len(changed)}"
        )
        facit_state = (
            "facit OFÖRÄNDRAT"
            if (facit_unexpected, facit_missing) == baseline_facit
            else f"facit +{facit_unexpected} −{facit_missing}"
        )
        print(f"{name}: {state}; {facit_state}")

    corrections = json.loads(
        args.headword_corrections.read_text(encoding="utf-8")
    )
    replacements = sum(
        source != target
        for source, target in corrections.items()
    )
    confirmations = len(corrections) - replacements
    insertions = len(facit.get("manual_insertions", []))
    print("\nMANUELLA SPECIALRÄTTNINGAR")
    print(f"Huvudordsersättningar: {replacements}")
    print(f"Uttryckliga bekräftelser: {confirmations}")
    print(f"Manuella lemma-infogningar: {insertions}")
    print(f"Totalt manuellt uppräknade poster: {len(corrections) + insertions}")


if __name__ == "__main__":
    main()
