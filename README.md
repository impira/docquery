<div align="center">

NOTE: DocQuery is not actively maintained anymore. We still welcome contributions and discussions among the community!

# DocQuery: Document Query Engine Powered by Large Language Models

[![Demo](https://img.shields.io/badge/Demo-Gradio-brightgreen)](https://huggingface.co/spaces/impira/docquery)
[![Demo](https://img.shields.io/badge/Demo-Colab-orange)](https://github.com/impira/docquery/blob/main/docquery_example.ipynb)
[![PyPI](https://img.shields.io/pypi/v/docquery?color=green&label=pip%20install%20docquery)](https://pypi.org/project/docquery/)
[![Discord](https://img.shields.io/discord/1015684761471160402?label=Chat)](https://discord.gg/HucNfTtx7V)
[![Downloads](https://static.pepy.tech/personalized-badge/docquery?period=total&units=international_system&left_color=grey&right_color=green&left_text=Downloads)](https://pepy.tech/project/docquery)

</div>

DocQuery is a library and command-line tool that makes it easy to analyze semi-structured and unstructured documents (PDFs, scanned
images, etc.) using large language models (LLMs). You simply point DocQuery at one or more documents and specify a
question you want to ask. DocQuery is created by the team at [Impira](https://impira.com?utm_source=github&utm_medium=referral&utm_campaign=docquery).

## Quickstart (CLI)

To install `docquery`, you can simply run `pip install docquery`. This will install the command line tool as well as the library.
If you want to run OCR on images, then you must also install the [tesseract](https://github.com/tesseract-ocr/tesseract) library:

- Mac OS X (using [Homebrew](https://brew.sh/)):

  ```sh
  brew install tesseract
  ```

- Ubuntu:

  ```sh
  apt install tesseract-ocr
  ```

`docquery` scan allows you to ask one or more questions to a single document or directory of files. For example, you can
find the invoice number <https://templates.invoicehome.com/invoice-template-us-neat-750px.png> with:

```bash
docquery scan "What is the invoice number?" https://templates.invoicehome.com/invoice-template-us-neat-750px.png
```

If you have a folder of documents on your machine, you can run something like

```bash
docquery scan "What is the effective date?" /path/to/contracts/folder
```

to determine the effective date of every document in the folder.

## Quickstart (Library)

DocQuery can also be used as a library. It contains two basic abstractions: (1) a `DocumentQuestionAnswering` pipeline
that makes it simple to ask questions of documents and (2) a `Document` abstraction that can parse various types of documents
to feed into the pipeline.

```python
>>> from docquery import document, pipeline
>>> p = pipeline('document-question-answering')
>>> doc = document.load_document("/path/to/document.pdf")
>>> for q in ["What is the invoice number?", "What is the invoice total?"]:
...     print(q, p(question=q, **doc.context))
```

## Use cases

DocQuery excels at a number of use cases involving structured, semi-structured, or unstructured documents. You can ask questions about
invoices, contracts, forms, emails, letters, receipts, and many more. You can also classify documents. We will continue evolving the model,
offer more modeling options, and expanding the set of supported documents. We welcome feedback, requests, and of course contributions to
help achieve this vision.

## How it works

Under the hood, docquery uses a pre-trained zero-shot language model, based on [LayoutLM](https://arxiv.org/abs/1912.13318), that has been
fine-tuned for a question-answering task. The model is trained using a combination of [SQuAD2.0](https://rajpurkar.github.io/SQuAD-explorer/)
and [DocVQA](https://rrc.cvc.uab.es/?ch=17) which make it particularly well suited for complex visual question answering tasks on
a wide variety of documents. The underlying model is also published on HuggingFace as [impira/layoutlm-document-qa](https://huggingface.co/impira/layoutlm-document-qa)
which you can access directly.

## Limitations

DocQuery is intended to have a small install footprint and be simple to work with. As a result, it has some limitations:

- Models must be pre-trained. Although DocQuery uses a zero-shot model that can adapt based on the question you provide, it does not learn from your data.
- Support for images and PDFs. Currently DocQuery supports images and PDFs, with or without embedded text. It does not support word documents, emails, spreadsheets, etc.
- Scalar text outputs. DocQuery only produces text outputs (answers). It does not support richer scalar types (i.e. it treats numbers and dates as strings) or tables.

## Advanced features

### Using Donut 🍩

If you'd like to test `docquery` with [Donut](https://arxiv.org/abs/2111.15664), you must install the required extras:

```bash
pip install docquery[donut]
```

You can then run

```bash
docquery scan "What is the effective date?" /path/to/contracts/folder --checkpoint 'naver-clova-ix/donut-base-finetuned-docvqa'
```

### Classifying documents

To classify documents, you simply add the `--classify` argument to `scan`. You can specify any [image classification](https://huggingface.co/models?pipeline_tag=image-classification&sort=downloads)
model on Hugging Face's hub. By default, the classification pipeline uses [Donut](https://huggingface.co/spaces/nielsr/donut-rvlcdip) (which requires
the installation instructions above):

```bash

# Classify documents
docquery scan --classify  /path/to/contracts/folder --checkpoint 'naver-clova-ix/donut-base-finetuned-docvqa'

# Classify documents and ask a question too
docquery scan --classify "What is the effective date?" /path/to/contracts/folder --checkpoint 'naver-clova-ix/donut-base-finetuned-docvqa'
```

### Scraping webpages

DocQuery can read files through HTTP/HTTPs out of the box. However, if you want to read HTML documents, you can do that too by installing the
`[web]` extension. The extension uses the [webdriver-manager](https://pypi.org/project/webdriver-manager/) library which can install a Chrome
driver on your system automatically, but you'll need to make sure Chrome is installed globally.

```
# Find the top post on hacker news
docquery scan "What is the #1 post's title?" https://news.ycombinator.com
```

## Where to go from here

DocQuery is a swiss army knife tool for working with documents and experiencing the power of modern machine learning. You can use it
just about anywhere, including behind a firewall on sensitive data, and test it with a wide variety of documents. Our hope is that
DocQuery enables many creative use cases for document understanding by making it simple and easy to ask questions from your documents.

When you run DocQuery for the first time, it will download some files (e.g. the models and some library code from HuggingFace). However,
nothing leaves your computer -- the OCR is done locally, models run locally, etc. This comes with the benefit of security and privacy;
however, it comes at the cost of runtime performance and some accuracy.

If you find yourself wondering how to achieve higher accuracy, work with more file types, teach the model with your own data, have
a human-in-the-loop workflow, or query the data you're extracting, then do not fear -- you are running into the challenges that
every organization does while putting document AI into production. The [Impira](https://www.impira.com/) platform is designed to
solve these problems in an easy and intuitive way. Impira comes with a QA model that is additionally trained on proprietary datasets
and can achieve 95%+ accuracy out-of-the-box for most use cases. It also has an intuitive UI that enables subject matter experts to label
and improve the models, as well as an API that makes integration a breeze. Please [sign up for the product](https://www.impira.com/signup) or
[reach out to us](info@impira.com) for more details.

## Status

DocQuery is a new project. Although the underlying models are running in production, we've just recently released our code in open source
and are actively working with the OSS community to upstream some of the changes we've made (e.g. [the model](https://github.com/huggingface/transformers/pull/18407)
and [pipeline](https://github.com/huggingface/transformers/pull/18414)). DocQuery is rapidly changing, and we are likely to make breaking
API changes. If you would like to run it in production, then we suggest pinning a version or commit hash. Either way, please get in touch
with us at [oss@impira.com](mailto:oss@impira.com) with any questions or feedback.

## Acknowledgements

DocQuery would not be possible without the contributions of many open source projects:

- [pdfplumber](https://github.com/jsvine/pdfplumber) / [pdfminer.six](https://github.com/pdfminer/pdfminer.six)
- [Pillow](https://pillow.readthedocs.io/en/stable/)
- [pytorch](https://pytorch.org/)
- [tesseract](https://github.com/tesseract-ocr/tesseract) / [pytesseract](https://pypi.org/project/pytesseract/)
- [transformers](https://github.com/impira/transformers)

and many others!

## License

This project is licensed under the [MIT license](LICENSE).

It contains code that is copied and adapted from transformers (<https://github.com/huggingface/transformers>),
which is [Apache 2.0 licensed](http://www.apache.org/licenses/LICENSE-2.0). Files containing this code have
been marked as such in their comments.
