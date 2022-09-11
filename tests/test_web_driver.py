from docquery.web import get_webdriver


def test_singleton():
    d1 = get_webdriver()
    d2 = get_webdriver()
    assert d1 is d2, "Both webdrivers should map to the same instance"


def test_readme_file():
    driver = get_webdriver()
    driver.get("https://github.com/impira/docquery/blob/ef73fa7e8069773ace03efae2254f3a510a814ef/README.md")
    word_boxes = driver.find_word_boxes()

    # This sanity checks the logic that merges word boxes
    assert len(word_boxes["word_boxes"]) > 20, "Expect multiple word boxes"

    # Make sure the last screenshot is shorter than the previous ones
    _, screenshots = driver.scroll_and_screenshot()
    assert len(screenshots) > 1, "Expect multiple pages"
    assert (
        screenshots[0].size[1] - screenshots[-1].size[1] > 10
    ), "Expect the last page to be shorter than the first several"
