from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path

from PIL import Image


BASE = Path(__file__).with_name("debug_runeberg_ocr_base.py")


def _load_base_module():
    spec = importlib.util.spec_from_file_location("debug_runeberg_ocr_base", BASE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Kunde inte läsa {BASE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _guides_html(x_models: dict, image_width: int) -> str:
    result: list[str] = []
    definitions = (
        ("homonym", "H", "Homonymindex", "#7c3aed"),
        ("article", "A", "Artikelstart", "#16a34a"),
        ("continuation", "F", "Fortsättningsrad", "#ea580c"),
    )
    for column in (1, 2):
        model = x_models[column]
        fallback = model.article_x - max(10.0, abs(model.continuation_x - model.article_x) * 0.65)
        positions = {
            "homonym": model.homonym_x if model.homonym_x is not None else fallback,
            "article": model.article_x,
            "continuation": model.continuation_x,
        }
        for kind, short, label, color in definitions:
            x = max(0.0, min(float(image_width), float(positions[kind])))
            estimated = kind == "homonym" and model.homonym_x is None
            result.append(
                '<div class="x-guide x-guide-%s" data-x="%.3f" style="--guide-color:%s">'
                '<span class="x-guide-label">%s%d · x=%.1f%s</span>'
                '<span class="x-guide-label x-guide-label-middle">%s%d</span>'
                '</div>'
                % (
                    kind,
                    x,
                    color,
                    short,
                    column,
                    x,
                    " uppsk." if estimated else "",
                    short,
                    column,
                )
            )
    return "".join(result)


def main() -> None:
    module = _load_base_module()
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
        html = original_review_html(
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

        css = """
<style>
.x-guide {
    position: absolute;
    top: 0;
    bottom: 0;
    width: 0;
    z-index: 30;
    border-left: 6px solid var(--guide-color);
    pointer-events: none;
    opacity: .92;
    filter: drop-shadow(0 0 2px white) drop-shadow(0 0 3px black);
}
.x-guide-homonym { border-left-style: dotted; }
.x-guide-article { border-left-style: solid; }
.x-guide-continuation { border-left-style: dashed; }
.x-guide-label {
    position: absolute;
    top: 8px;
    left: 8px;
    padding: 4px 7px;
    border: 2px solid white;
    border-radius: 5px;
    background: var(--guide-color);
    color: white;
    font: 800 14px/1.15 system-ui, sans-serif;
    white-space: nowrap;
    box-shadow: 0 1px 5px #000b;
}
.x-guide-label-middle { top: 48%; font-size: 16px; }
.marker { z-index: 40; }
.guide-legend { margin-left: 14px; font-weight: 700; }
.guide-legend .h { color: #7c3aed; }
.guide-legend .a { color: #16a34a; }
.guide-legend .f { color: #ea580c; }
</style>
"""
        legend = (
            '<span class="guide-legend">'
            '<span class="h">H = homonym</span> · '
            '<span class="a">A = artikelstart</span> · '
            '<span class="f">F = fortsättning</span>'
            '</span>'
        )
        guides = _guides_html(x_models, image_width)
        html = html.replace("</head>", css + "</head>", 1)
        html = html.replace(
            '<span id="zoomLabel">100 %</span>',
            '<span id="zoomLabel">100 %</span>' + legend,
            1,
        )
        html = html.replace(
            '<div class="marker" id="marker"></div>',
            guides + '<div class="marker" id="marker"></div>',
            1,
        )
        html = html.replace(
            "const rows=[...document.querySelectorAll('#articleRows tr[data-index]')];let zoom=1,selected=-1;",
            "const rows=[...document.querySelectorAll('#articleRows tr[data-index]')],guides=[...document.querySelectorAll('.x-guide')];let zoom=1,selected=-1;",
            1,
        )
        html = html.replace(
            "function setZoom(v){zoom=Math.max(.15,Math.min(3,v));scanStage.style.width=`${naturalWidth*zoom}px`;scanStage.style.height=`${naturalHeight*zoom}px`;scanImage.style.width=`${naturalWidth*zoom}px`;scanImage.style.height=`${naturalHeight*zoom}px`;document.getElementById('zoomLabel').textContent=`${Math.round(zoom*100)} %`;if(selected>=0)positionMarker(rows[selected])}",
            "function setZoom(v){zoom=Math.max(.15,Math.min(3,v));scanStage.style.width=`${naturalWidth*zoom}px`;scanStage.style.height=`${naturalHeight*zoom}px`;scanImage.style.width=`${naturalWidth*zoom}px`;scanImage.style.height=`${naturalHeight*zoom}px`;guides.forEach(g=>g.style.left=`${(+g.dataset.x)*zoom}px`);document.getElementById('zoomLabel').textContent=`${Math.round(zoom*100)} %`;if(selected>=0)positionMarker(rows[selected])}",
            1,
        )
        return html

    module._review_html = review_html_with_guides
    module.main()


if __name__ == "__main__":
    main()
