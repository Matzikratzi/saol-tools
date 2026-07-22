from __future__ import annotations

"""Run the Runeberg OCR debugger with a per-printed-line rule analysis.

Usage:
    PYTHONPATH=. python3 scripts/debug_runeberg_ocr_rows.py 20 --html --open

This wrapper leaves the normal debugger unchanged.  It records every printed
line after geometric classification and adds a detailed table to the HTML
report so that the A/F/H rules can be reviewed row by row.
"""

import statistics

import debug_runeberg_ocr as debug


_DIAGNOSTICS: list[dict[str, object]] = []


def _explanation(module, line, model, candidate: bool) -> tuple[str, str]:
    lexical = bool(module.WORD_RE.search(line.first.text))
    word_x = float(line.letter_start_x)
    raw_x = float(line.raw_start_x)
    article_x = float(model.article_x)
    continuation_x = float(model.continuation_x)
    boundary_x = float(model.boundary_x)

    if not lexical:
        decision = "inte lexikal rad"
    elif line.x_class.endswith("article"):
        decision = "artikelstart enligt nuvarande regel"
    else:
        decision = "fortsättningsrad enligt nuvarande regel"

    reasons: list[str] = []
    if word_x < article_x:
        reasons.append(
            f"ordstarten ligger {article_x - word_x:.1f} px vänster om A"
        )
    elif word_x > article_x:
        reasons.append(
            f"ordstarten ligger {word_x - article_x:.1f} px höger om A"
        )
    else:
        reasons.append("ordstarten ligger exakt på A")

    if word_x <= boundary_x:
        reasons.append(
            f"x={word_x:.1f} är vänster om A/F-gränsen {boundary_x:.1f}, därför A-klassen"
        )
    else:
        reasons.append(
            f"x={word_x:.1f} är höger om A/F-gränsen {boundary_x:.1f}, därför F-klassen"
        )

    if candidate:
        if line.has_homonym_marker:
            reasons.append(
                f"prefixgeometrin godtogs som H; rå-x={raw_x:.1f} och ord-x={word_x:.1f}"
            )
        else:
            reasons.append(
                "prefixgeometrin såg ut som en möjlig H-markör men underkändes av positionsreglerna"
            )
    elif line.has_homonym_marker:
        reasons.append("raden är markerad som H trots att kontrollkörningen inte hittar en kandidat")

    if lexical:
        reasons.append(f"fetstilspoäng={line.bold_score:.2f}")
    else:
        reasons.append("ingen bokstav hittades i första ordobjektet")

    reasons.append(
        f"avstånd: ΔA={word_x - article_x:+.1f}, ΔF={word_x - continuation_x:+.1f} px"
    )
    return decision, "; ".join(reasons)


def _analysis_html(module) -> str:
    rows: list[str] = []
    for item in _DIAGNOSTICS:
        rows.append(
            "<tr>"
            f"<td>{item['column']}</td>"
            f"<td>{item['top']:.0f}</td>"
            f"<td><code>{module._escape(item['text'])}</code></td>"
            f"<td>{item['raw_x']:.1f}</td>"
            f"<td>{item['word_x']:.1f}</td>"
            f"<td>{item['delta_a']:+.1f}</td>"
            f"<td>{item['delta_f']:+.1f}</td>"
            f"<td>{'ja' if item['candidate'] else ''}</td>"
            f"<td>{'ja' if item['accepted_h'] else ''}</td>"
            f"<td>{module._escape(item['x_class'])}</td>"
            f"<td>{item['bold_score']:.2f}</td>"
            f"<td><strong>{module._escape(item['decision'])}</strong><br>"
            f"{module._escape(item['reason'])}</td>"
            "</tr>"
        )

    return """
<style>
.line-analysis { margin-top:18px; }
.line-analysis .table-wrap { height:auto; max-height:none; overflow:auto; }
.line-analysis table { min-width:1500px; }
.line-analysis td:nth-child(3), .line-analysis td:last-child { white-space:normal; }
.line-analysis code { white-space:pre-wrap; }
</style>
<section class="panel line-analysis">
<h2>Radanalys – geometriregler för varje tryckt rad</h2>
<div class="table-wrap"><table>
<thead><tr>
<th>Spalt</th><th>y</th><th>OCR-rad</th><th>rå-x</th><th>ord-x</th>
<th>ΔA</th><th>ΔF</th><th>H-kandidat</th><th>H godtagen</th>
<th>x-klass</th><th>fet</th><th>Beslut och regelspår</th>
</tr></thead>
<tbody>""" + "".join(rows) + """</tbody></table></div></section>
"""


_original_geometry_build_lines = debug._geometry_build_lines


def _geometry_with_diagnostics(
    module, original_build_lines, observations, image_width, image_height
):
    lines, models = _original_geometry_build_lines(
        module, original_build_lines, observations, image_width, image_height
    )

    heights = [item.height for line in lines for item in line.items]
    median_height = statistics.median(heights) if heights else 1.0
    _DIAGNOSTICS.clear()

    for line in lines:
        model = models[line.column]
        _word_x, _word_object, _stripped, candidate = debug._prefix_geometry(
            module, line, median_height
        )
        decision, reason = _explanation(module, line, model, candidate)
        _DIAGNOSTICS.append(
            {
                "column": line.column,
                "top": float(line.top),
                "text": line.text,
                "raw_x": float(line.raw_start_x),
                "word_x": float(line.letter_start_x),
                "delta_a": float(line.letter_start_x - model.article_x),
                "delta_f": float(line.letter_start_x - model.continuation_x),
                "candidate": bool(candidate),
                "accepted_h": bool(line.has_homonym_marker),
                "x_class": line.x_class,
                "bold_score": float(line.bold_score),
                "decision": decision,
                "reason": reason,
            }
        )

    return lines, models


debug._geometry_build_lines = _geometry_with_diagnostics

_original_load_base_module = debug._load_base_module


def _load_base_module_with_analysis():
    module = _original_load_base_module()
    original_review_html = module._review_html

    def review_html_with_analysis(*args, **kwargs):
        report = original_review_html(*args, **kwargs)
        analysis = _analysis_html(module)
        return report.replace("</main>", analysis + "</main>", 1)

    module._review_html = review_html_with_analysis
    return module


debug._load_base_module = _load_base_module_with_analysis


if __name__ == "__main__":
    debug.main()
