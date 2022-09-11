import os
from pathlib import Path

from .config import get_logger
from .ext.functools import cached_property


log = get_logger("web")

try:
    from chromedriver_py import (  # This library allows you to download the driver for your version of chrome
        binary_path,
    )
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
        # Assumes the driver has been pointed at the right website already
        return self.driver.execute_script(
            self.lib_js
            + """
                return computeBoundingBoxes(document.body);
            """
        )

    # TODO: Handle horizontal scrolling
    def screenshots_png(self):
        tops = []
        images = []
        dims = self.driver.execute_script(
            self.lib_js
            + """
            return computeViewport()
        """
        )

        view_height = dims["vh"]
        doc_height = dims["dh"]

        self.driver.execute_script("window.scroll(0, 0)")
        while True:
            curr = self.driver.execute_script("return window.scrollY")
            tops.append(curr)
            images.append(self.driver.get_screenshot_as_png())
            if curr + view_height < doc_height:
                curr = self.driver.execute_script(f"window.scroll(0, {curr+view_height})")
            else:
                break

        return tops, images

    @cached_property
    def lib_js(self):
        with open(dir_path / "find_leaf_nodes.js", "r") as f:
            return f.read()


def get_webdriver():
    global WEB_DRIVER
    if WEB_DRIVER is None:
        WEB_DRIVER = WebDriver()
    return WEB_DRIVER
