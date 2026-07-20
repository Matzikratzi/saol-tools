from app.runeberg import extract_bold_headwords, page_urls


def test_page_urls():
    html, image = page_urls(19)
    assert html.endswith("/0019.html")
    assert image.endswith("/0019.3.png")


def test_extracts_only_bold_groups_from_hocr():
    hocr = """
    <html><body>
      <span class='ocr_line'>
        <span class='ocrx_word'><strong>Abborre</strong></span>
        <span class='ocrx_word'>-n</span>
        <span class='ocrx_word'><strong>-fiske</strong></span>
        <span class='ocrx_word'>förklaring</span>
      </span>
      <span class='ocr_line'>
        <span class='ocrx_word'><strong>efter</strong></span>
        <span class='ocrx_word'><strong>hand</strong></span>
        <span class='ocrx_word'>adv.</span>
      </span>
    </body></html>
    """
    assert extract_bold_headwords(hocr) == ["Abborre", "-fiske", "efter hand"]


def test_extra_bold_and_semibold_have_same_meaning():
    hocr = """
    <span class='ocr_line'>
      <span class='ocrx_word'><b>Abakus</b></span>
      <span class='ocrx_word'>-en</span>
      <span class='ocrx_word'><strong>abandon</strong></span>
    </span>
    """
    assert extract_bold_headwords(hocr) == ["Abakus", "abandon"]
