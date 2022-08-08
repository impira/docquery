all: build

VERSION=$(shell python  -c 'from src.docqa.version import VERSION; print(VERSION)')

build:
	python3 -m build

publish: build
	python3 -m twine upload dist/docqa-${VERSION}*

clean:
	rm -rf dist/*

develop:
	python3 -m venv venv
	bash -c 'source venv/bin/activate && python -m pip install --upgrade pip setuptools'
	bash -c 'source venv/bin/activate && python -m pip install -e .[dev]'
	@echo 'Run "source venv/bin/activate" to enter development mode'
