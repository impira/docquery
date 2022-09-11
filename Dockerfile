FROM python:3.10-slim-bullseye

RUN apt-get update \
  && apt-get install -y software-properties-common  \
  && add-apt-repository -y ppa:alex-p/tesseract-ocr5  \
  && apt-get install -y tesseract-ocr  \
  && apt-get install -y poppler-utils \
  && pip3 --no-cache-dir install --upgrade pip \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /project
COPY ["README.md", "pyproject.toml", "setup.py", "./"]
COPY ["src/", "./src"]

RUN pip install .
CMD ["python3"]