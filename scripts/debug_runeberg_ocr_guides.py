from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path

from PIL import Image


SOURCE = Path(__file__).with_name("debug_runeberg_ocr.py")


def _load_debug_module():
    spec = importlib.util.spec_from_file_location("debug_runeberg_ocr_base", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Kunde inte läsa {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _guide_html(x_models: dict, image_width: int) -> str:
    guides: list[str] = []
    colors = {
        "homonym": "#7c3aed",
        "article": "#059669",
        "continuation": "#d97706",
    }
    labels = {
        "homonym": "Homonym",
        "article": "Artikelstart",
        "continuation": "Fortsättning",
    }

    for column in (1, 2):
        model = x_models[column]
        fallback_gap = max(8.0, abs(model.continuation_x - model.article_x) * 0.55)
        positions = {
            "homonym": model.homonym_x if model.homonym_x is not None else model.article_x - fallback_gap,
            "article": model.article_x,
            "continuation": model.continuation_x,
        }
        for kind, x_value in positions.items():
            x_value = max(0.0, min(float(image_width), float(x_value)))
            left_percent = 100.0 * x_value / max(1, image_width)
            estimated = kind == "homonym" and model.homonym_x is None
            suffix = " (uppsk.)" if estimated else ""
            guides.append(
                '<div class="x-guide x-guide-%s" style="left:%.6f%%;--guide-color:%s">'
                '<span>Spalt %d · %s%s · x=%.1f</span></div>'
                % (
                    kind,
                    left_percent,
                    colors[kind],
                    column,
                    labels[kind],
                    suffix,
                    x_value,
                )
            )
    return "".join(guides)


def main() -> None:
    module = _load_debug_module()
    original_review_html = module._review_html

    def review_html_with_guides(
        page,
        source_url,
        image_content,
        raw_text,
        runeberg_lines,
        observations,
        articles,
        pairs,
        x_models,
        threshold,
        skew_degrees,
    ):
        result = original_review_html(
            page,
            source_url,
            image_content,
            raw_text,
            runeberg_lines,
            observations,
            articles,
            pairs,
            x_models,
            threshold,
            skew_degrees,
        )
        with Image.open(io.BytesIO(image_content)) as image:
            image_width = image.width

        guide_css = """
<style>
.x-guide {
    position: absolute;
    top: 0;
    bottom: 0;
    width: 0;
    z-index: 4;
    border-left: 3px solid var(--guide-color);
    pointer-events: none;
    filter: drop-shadow(0 0 1px white) drop-shadow(0 0 2px black);
}
.x-guide-homonym { border-left-style: dotted; }
.x-guide-article { border-left-style: solid; }
.x-guide-continuation { border-left-style: dashed; }
.x-guide span {
    position: sticky;
    top: 6px;
    display: inline-block;
    transform: translateX(5px);
    padding: 3px 5px;
    border-radius: 4px;
    background: color-mix(in srgb, var(--guide-color) 88%, black);
    color: white;
    font: 700 12px/1.15 system-ui, sans-serif;
    white-space: nowrap;
    box-shadow: 0 1px 4px #0008;
}
.marker { z-index: 8; }
.guide-legend {
    display: inline-flex;
    flex-wrap: wrap;
    gap: 9px;
    margin-left: 12px;
    font-size: .82rem;
}
.guide-key::before {
    content: "";
    display: inline-block;
    width: 18px;
    margin-right: 4px;
    vertical-align: middle;
    border-top: 3px solid currentColor;
}
.guide-key.homonym { color: #7c3aed; }
.guide-key.homonym::before { border-top-style: dotted; }
.guide-key.article { color: #059669; }
.guide-key.continuation { color: #d97706; }
.guide-key.continuation::before { border-top-style: dashed; }
</style>
"""
        legend = (
            '<span class="guide-legend">'
            '<span class="guide-key homonym">Homonym</span>'
            '<span class="guide-key article">Artikelstart</span>'
            '<span class="guide-key continuation">Fortsättning</span>'
            '</span>'
        )
        guides = _guide_html(x_models, image_width)

        result = result.replace("</head>", guide_css + "</head>", 1)
        result = result.replace(
            '<span id="zoomLabel">100 %</span>',
            '<span id="zoomLabel">100 %</span>' + legend,
            1,
        )
        result = result.replace(
            '<div class="marker" id="marker"></div>',
            guides + '<div class="marker" id="marker"></div>',
            1,
        )
        return result

    module._review_html = review_html_with_guides
    module.main()


if __name__ == "__main__":
    main()
