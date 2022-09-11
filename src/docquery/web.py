import os
from io import BytesIO
from pathlib import Path

from PIL import Image

from .config import get_logger
from .ext.functools import cached_property


log = get_logger("web")

try:
    from selenium import webdriver
    from selenium.common import exceptions
    from selenium.webdriver.chrome.options import Options
    from webdriver_manager.chrome import ChromeDriverManager
    from webdriver_manager.core.utils import ChromeType

    WEB_AVAILABLE = True
except ImportError:
    WEB_AVAILABLE = False


FIND_LEAF_NODES_JS = None
WEB_DRIVER = None
dir_path = Path(os.path.dirname(os.path.realpath(__file__)))


class WebDriver:
    def __init__(self):
        if not WEB_AVAILABLE:
            raise ValueError(
                "Web imports are unavailable. You must install the [web] extra and chrome or" " chromium system-wide."
            )

        self._reinit_driver()

    def _reinit_driver(self):
        options = Options()
        options.headless = True
        options.add_argument("--window-size=1920,1200")
        if os.geteuid() == 0:
            options.add_argument("--no-sandbox")

        self.driver = webdriver.Chrome(
            options=options, executable_path=ChromeDriverManager(chrome_type=ChromeType.CHROMIUM).install()
        )

    def get(self, page, retry=True):
        try:
            self.driver.get(page)
        except exceptions.InvalidSessionIdException:
            if retry:
                # Forgive an invalid session once and try again
                self._reinit_driver()
                return self.get(page, retry=False)
            else:
                raise

    def get_html(self, html):
        # https://stackoverflow.com/questions/22538457/put-a-string-with-html-javascript-into-selenium-webdriver
        self.get("data:text/html;charset=utf-8," + html)

    def find_word_boxes(self):
        # Assumes the driver has been pointed at the right website already
        return self.driver.execute_script(
            self.lib_js
            + """
                return computeBoundingBoxes(document.body);
            """
        )

    # TODO: Handle horizontal scrolling
    def scroll_and_screenshot(self):
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

        try:
            self.driver.execute_script("window.scroll(0, 0)")
            curr = self.driver.execute_script("return window.scrollY")

            while True:
                tops.append(curr)
                images.append(Image.open(BytesIO(self.driver.get_screenshot_as_png())))
                if curr + view_height < doc_height:
                    self.driver.execute_script(f"window.scroll(0, {curr+view_height})")

                curr = self.driver.execute_script("return window.scrollY")
                if curr <= tops[-1]:
                    break
        finally:
            # Reset scroll to the top of the page
            self.driver.execute_script("window.scroll(0, 0)")

        if len(tops) >= 2:
            _, second_last_height = images[-2].size
            if tops[-1] - tops[-2] < second_last_height:
                # This occurs when the last screenshot should be "clipped". Adjust the last "top"
                # to correspond to the right view_height and clip the screenshot accordingly
                delta = second_last_height - (tops[-1] - tops[-2])
                tops[-1] += delta

                last_img = images[-1]
                last_width, last_height = last_img.size
                images[-1] = last_img.crop((0, delta, last_width, last_height))

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
