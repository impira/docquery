FROM ubuntu:latest

RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && apt-get install -y tesseract-ocr  \
  && apt-get install -y poppler-utils \
  && apt-get install -y  git \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 --no-cache-dir install --upgrade pip \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /project
COPY ["README.md", "pyproject.toml", "setup.py", "./"]
COPY ["src/", "./src"]

RUN pip install .[donut]
CMD ["python3"]