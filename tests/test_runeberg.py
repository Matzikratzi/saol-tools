from app.runeberg import extract_ocr, page_urls


def test_page_urls():
    html, image = page_urls(19)
    assert html.endswith("/0019.html")
    assert image.endswith("/0019.3.png")


def test_extract_ocr_from_pre():
    html = '<html><body><p>Below is the raw OCR text</p><pre>19\nabbé -n -er\n</pre></body></html>'
    assert extract_ocr(html) == "19\nabbé -n -er"


def test_extract_ocr_when_marker_is_split_by_link():
    html = """
    <html><body>
      <p>Below is the raw OCR text from the above scanned image.
         Do you see an error? <a href='/proof'>Proofread the page now!</a>
         Här nedan syns maskintolkade texten från faksimilbilden ovan.
      </p>
      <p>This page has never been proofread. / Denna sida har aldrig korrekturlästs.</p>
      <p>19</p>
      <p>abakus -en -er</p>
      <p>abandon -en</p>
      <p>&lt;&lt; prev. page &lt;&lt;</p>
    </body></html>
    """
    assert extract_ocr(html) == "19\nabakus -en -er\nabandon -en"
