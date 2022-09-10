import os
from pathlib import Path

from .config import get_logger


log = get_logger("web")

try:
    from chromedriver_py import binary_path  # This library allows you to download the driver for your version of chrome
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options

    WEB_AVAILABLE = True
except ImportError as e:
    log.warning("%s" % (e))
    WEB_AVAILABLE = False


FIND_LEAF_NODES_JS = None
WEB_DRIVER = None
dir_path = Path(os.path.dirname(os.path.realpath(__file__)))


class WebDriver:
    def __init__(self):
        if not WEB_AVAILABLE:
            raise ValueError(
                "Web imports are unavailable. You must install the [web] extra and a version of"
                " chromedriver_py compatible with your system"
            )

        options = Options()
        options.headless = True
        options.add_argument("--window-size=1920,1200")

        self.driver = webdriver.Chrome(options=options, executable_path=binary_path)

    def get(self, page):
        self.driver.get(page)

    def get_html(self, html):
        # https://stackoverflow.com/questions/22538457/put-a-string-with-html-javascript-into-selenium-webdriver
        self.driver.get("data:text/html;charset=utf-8," + html)

    def find_word_boxes(self):
        global FIND_LEAF_NODES_JS
        if FIND_LEAF_NODES_JS is None:
            with open(dir_path / "find_leaf_nodes.js", "r") as f:
                FIND_LEAF_NODES_JS = (
                    f.read()
                    + """
                    return computeBoundingBoxes(document.body);
                """
                )
        # Assumes the driver has been pointed at the right website already
        return self.driver.execute_script(FIND_LEAF_NODES_JS)

    def screenshots_png(self):
        # TODO: I think this will only return the first "pane" as a screenshot. We should probably adjust
        # the horizontal size and scroll vertically to generate multiple previews
        return [self.driver.get_screenshot_as_png()]


def get_webdriver():
    global WEB_DRIVER
    if WEB_DRIVER is None:
        WEB_DRIVER = WebDriver()
    return WEB_DRIVER
