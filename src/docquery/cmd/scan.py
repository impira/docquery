import os
import pathlib

from .. import pipeline
from ..config import get_logger
from ..document import UnsupportedDocument, load_document
from ..ocr_reader import OCR_MAPPING
from ..transformers_patch import PIPELINE_DEFAULTS


log = get_logger("scan")


def build_parser(subparsers, parent_parser):
    parser = subparsers.add_parser(
        "scan",
        help="Scan a directory and ask one or more questions of the documents in it.",
        parents=[parent_parser],
    )

    parser.add_argument(
        "questions", default=[], nargs="*", type=str, help="One or more questions to ask of the documents"
    )

    parser.add_argument("path", type=str, help="The file or directory to scan")

    parser.add_argument(
        "--ocr", choices=list(OCR_MAPPING.keys()), default=None, help="The OCR engine you would like to use"
    )
    parser.add_argument(
        "--classify",
        default=False,
        action="store_true",
        help="Classify documents while scanning them",
    )
    parser.add_argument(
        "--classify-checkpoint",
        default=None,
        help=f"A custom model checkpoint to use (other than {PIPELINE_DEFAULTS['image-classification']})",
    )

    parser.set_defaults(func=main)
    return parser


def main(args):
    paths = []
    if pathlib.Path(args.path).is_dir():
        for root, dirs, files in os.walk(args.path):
            for fname in files:
                paths.append(pathlib.Path(root) / fname)
    else:
        paths.append(args.path)

    docs = []
    for p in paths:
        try:
            docs.append((p, load_document(str(p), ocr_reader=args.ocr)))
            log.info(f"Loading {p}")
        except UnsupportedDocument as e:
            log.warning(f"Cannot load {p}: {e}. Skipping...")

    log.info("Done loading files. Loading pipeline...")
    nlp = pipeline("document-question-answering", model=args.checkpoint)
    log.info("Ready to start evaluating!")

    if args.classify:
        classify = pipeline("image-classification", model=args.classify_checkpoint)

    max_fname_len = max(len(str(p)) for (p, d) in docs)
    max_question_len = max(len(q) for q in args.questions) if len(args.questions) > 0 else 0
    for i, (p, d) in enumerate(docs):
        if i > 0 and len(args.questions) > 1:
            print("")
        if args.classify:
            cls = classify(images=d.preview[0])[0]
            print(f"{str(p):<{max_fname_len}} Document Type: {cls['label']}")

        for q in args.questions:
            try:
                response = nlp(question=q, **d.context)
                if isinstance(response, list):
                    response = response[0] if len(response) > 0 else None
            except Exception:
                log.error(f"Failed while processing {str(p)} on question: '{q}'")
                raise

            answer = response["answer"] if response is not None else "NULL"
            print(f"{str(p):<{max_fname_len}} {q:<{max_question_len}}: {answer}")
