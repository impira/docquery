import os

import setuptools

dir_name = os.path.abspath(os.path.dirname(__file__))

version_contents = {}
with open(os.path.join(dir_name, "src", "docquery", "version.py"), encoding="utf-8") as f:
    exec(f.read(), version_contents)

with open(os.path.join(dir_name, "README.md"), "r", encoding="utf-8") as f:
    long_description = f.read()

install_requires = [
    "torch >= 1.0",
    "pdf2image",
    "pdfplumber",
    "Pillow",
    "pydantic",
    "pytesseract",  # TODO: Test what happens if the host machine does not have tesseract installed
    "requests",
    "easyocr",
    "transformers >= 4.23",
]
extras_require = {
    "dev": [
        "black",
        "build",
        "flake8",
        "flake8-isort",
        "isort==5.10.1",
        "pre-commit",
        "pytest",
        "twine",
    ],
    "donut": [
        "sentencepiece",
        "protobuf<=3.20.1",
    ],
    "web": [
        "selenium",
        "webdriver-manager",
    ],
    "cli": [],
}
extras_require["all"] = sorted({package for packages in extras_require.values() for package in packages})

setuptools.setup(
    name="docquery",
    version=version_contents["VERSION"],
    author="Impira Engineering",
    author_email="engineering@impira.com",
    description="DocQuery: An easy way to extract information from documents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/impira/docquery",
    project_urls={
        "Bug Tracker": "https://github.com/impira/docquery/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    package_data={"": ["find_leaf_nodes.js"]},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.7.0",
    entry_points={
        "console_scripts": ["docquery = docquery.cmd.__main__:main"],
    },
    install_requires=install_requires,
    extras_require=extras_require,
)
