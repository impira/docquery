FROM python:3

RUN apt-get update && apt-get upgrade -y && \
    pip install docquery && \
    apt-get install -qq -y tesseract-ocr poppler-utils

WORKDIR /docquery/src/docquery

# Add a script to be executed every time the container starts.
COPY entrypoint.sh /usr/bin/
RUN chmod +x /usr/bin/entrypoint.sh
ENTRYPOINT ["entrypoint.sh"]
