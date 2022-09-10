import os
from pathlib import Path

from chromedriver_py import binary_path  # This library allows you to download the driver for your version of chrome
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By


FIND_LEAF_NODES_JS = None
dir_path = Path(os.path.dirname(os.path.realpath(__file__)))


def get_webdriver():
    options = Options()
    options.headless = True
    options.add_argument("--window-size=1920,1200")

    return webdriver.Chrome(options=options, executable_path=binary_path)


def find_word_boxes(driver):
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
    return driver.execute_script(FIND_LEAF_NODES_JS)
