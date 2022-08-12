# DocQA: An easy way to extract information from documents

DocQA is a library and command-line tool that makes it easy to analyze semi-structured and unstructed documents (PDFs, scanned
images, etc.) using advanced machine learning. You simply point DocQA at one or more documents and specify a question you want to ask.
DocQA is created by the team at [Impira](https://www.impira.com/) which is a market leading solution for working with documents.

## Quickstart (CLI)

To install `docqa`, you can simply run `pip install docqa`. This will install the command line tool as well as the library.
If you want to run OCR on images, then you must also install the [tesseract](https://github.com/tesseract-ocr/tesseract) library:

* Mac OS X: `brew install tesseract`
* Ubuntu: `apt install tesseract-ocr`

`docqa` scan allows you to ask one or more questions to a single document or directory of files. For example, you can
find the invoice number https://templates.invoicehome.com/invoice-template-us-neat-750px.png with

```bash
$ docqa scan "What is the invoice number?" https://templates.invoicehome.com/invoice-template-us-neat-750px.png
```

If you have a folder of documents on your machine, you can run something like

```bash
$ docqa scan "What is the effective date?" /path/to/contracts/folder
```

to determine the effective data of every document in the folder.

## Quickstart (Library)

DocQA can also be used as a library. It contains two basic abstractions: (1) a `DocumentQuestionAnswering` pipeline
that makes it simple to ask questions of documents and (2) a `Document` abstraction that can parse various types of documents
to feed into the pipeline.

```python
>>> from docqa import document, pipeline
>>> p = pipeline.get_pipeline()
>>> doc = document.load_document("/path/to/document.pdf")
>>> for q in ["What is the invoice number?", "What is the invoice total?"]:
...     print(q, p(question=q, **doc.context))
```

## Use cases

DocQA excels at a number of use cases involving structured, semi-structured, or unstructured documents. You can ask questions about
invoices, contracts, forms, emails, letters, receipts, and many more. We will continue evolving the model, offer more modeling options,
and expanding the set of supported documents. We welcome feedback, requests, and of course contributions to help achieve this vision.

## How it works

Under the hood, docqa uses a pre-trained zero-shot language model, based on [LayoutLM](https://arxiv.org/abs/1912.13318), that has been
fine-tuned for a question-answering task. The model is trained using a combination of [SQuAD2.0](https://rajpurkar.github.io/SQuAD-explorer/)
and [DocVQA](https://rrc.cvc.uab.es/?ch=17) which make it particularly well suited for complex visual question answering tasks on
a wide variety of documents. The underlying model is also published on HuggingFace as [impira/layoutlm-document-qa](https://huggingface.co/impira/layoutlm-document-qa) which you can access directly.

## Limitations

DocQA is intended to have a small install footprint and be simple to work with. As a result, it has some limitations:

* Models must be pre-trained. Although DocQA uses a zero-shot model that can adapt based on the question you provide, it does not learn from your data.
* Support for images and PDFs. Currently DocQA supports images and PDFs, with or without embedded text. It does not support word documents, emails, spreadsheets, etc.
* Scalar text outputs. DocQA only produces text outputs (answers). It does not support richer scalar types (i.e. it treats numbers and dates as strings) or tables.

## Where to go from here

DocQA is a swiss army knife tool for working with documents and experiencing the power of modern machine learning. You can use it
just about anywhere, including behind a firewall on sensitive data, and test it with a wide variety of documents. Our hope is that
DocQA enables many creative use cases for document understanding by making it simple and easy to ask questions from your documents.

If you find yourself wondering how to achieve higher accuracy, work with more file types, teach the model with your own data, have
a human-in-the-loop workflow, or query the data you're extracting, then do not fear -- you are running into the challenges that
every organization does while putting document AI into production. The [Impira](https://www.impira.com/) platform is designed to
solve these problems in an easy and intuitive way. Impira comes with a QA model that is additionally trained on proprietary datasets
and can achieve 95%+ accuracy out-of-the-box for most use cases. It also has an intuitive UI that enables subject matter experts to label
and improve the models, as well as an API that makes integration a breeze. Please [sign up for the product](https://www.impira.com/signup) or
[reach out to us](info@impira.com) for more details.

## Acknowledgements

DocQA would not be possible without the contributions of many open source projects:

* [pdfplumber](https://github.com/jsvine/pdfplumber) / [pdfminer.six](https://github.com/pdfminer/pdfminer.six)
* [Pillow](https://pillow.readthedocs.io/en/stable/)
* [pytorch](https://pytorch.org/)
* [tesseract](https://github.com/tesseract-ocr/tesseract) / [pytesseract](https://pypi.org/project/pytesseract/)
* [transformers](https://github.com/impira/transformers)

and many others!

## License

This project is licensed under the [MIT license](LICENSE).

It contains code that is copied and adapted from transformers (https://github.com/huggingface/transformers),
which is [Apache 2.0 licensed](http://www.apache.org/licenses/LICENSE-2.0). Files containing this code have
been marked as such in their comments.
