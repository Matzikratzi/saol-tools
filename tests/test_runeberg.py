from app.runeberg import extract_ocr, page_urls


def test_page_urls():
    html, image = page_urls(19)
    assert html.endswith("/0019.html")
    assert image.endswith("/0019.3.png")


def test_extract_ocr_from_pre():
    html = '<html><body><p>Below is the raw OCR text</p><pre>19\nabbé -n -er\n</pre></body></html>'
    assert extract_ocr(html) == "19\nabbé -n -er"
