import unittest

from PIL import Image, ImageDraw

from scripts.lodstreck_visual_audit import (
    Component,
    boundary_fraction,
    glyph_components,
    visual_scores,
)


class LodstreckVisualAuditTests(unittest.TestCase):
    def test_boundary_fraction_uses_structured_stem(self):
        self.assertAlmostEqual(boundary_fraction("agla", "ag|a"), 2 / 4)
        self.assertAlmostEqual(
            boundary_fraction("-sldrift", "-s|drift"), 1 / 7
        )

    def test_thin_deep_stroke_scores_above_letters(self):
        image = Image.new("L", (70, 40), 255)
        draw = ImageDraw.Draw(image)
        draw.rectangle((5, 8, 16, 31), fill=0)
        draw.rectangle((25, 8, 36, 31), fill=0)
        draw.rectangle((47, 10, 49, 37), fill=0)
        components = glyph_components(image)
        scores = visual_scores(components)
        deepest = max(
            range(len(components)), key=lambda index: components[index].bottom
        )
        self.assertEqual(deepest, max(range(len(scores)), key=scores.__getitem__))

    def test_dot_penalizes_i_like_stroke(self):
        plain = [
            Component(5, 8, 16, 32, 200),
            Component(29, 12, 32, 36, 60),
        ]
        dotted = [
            plain[0],
            Component(29, 12, 32, 36, 60, has_dot=True),
        ]
        self.assertLess(visual_scores(dotted)[1], visual_scores(plain)[1])


if __name__ == "__main__":
    unittest.main()
