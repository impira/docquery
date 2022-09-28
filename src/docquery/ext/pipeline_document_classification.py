# This file is copied from transformers:
#    https://github.com/huggingface/transformers/blob/bb6f6d53386bf2340eead6a8f9320ce61add3e96/src/transformers/pipelines/image_classification.py
# And has been modified to support Donut
import re
from typing import List, Optional, Tuple, Union

import torch
from transformers.pipelines.base import PIPELINE_INIT_ARGS, ChunkPipeline
from transformers.pipelines.text_classification import ClassificationFunction, sigmoid, softmax
from transformers.utils import ExplicitEnum, add_end_docstrings, logging

from .pipeline_document_question_answering import ImageOrName, apply_tesseract
from .qa_helpers import TESSERACT_LOADED, VISION_LOADED, load_image


logger = logging.get_logger(__name__)


class ModelType(ExplicitEnum):
    Standard = "standard"
    VisionEncoderDecoder = "vision_encoder_decoder"


def donut_token2json(tokenizer, tokens, is_inner_value=False):
    """
    Convert a (generated) token sequence into an ordered JSON format.
    """
    output = dict()

    while tokens:
        start_token = re.search(r"<s_(.*?)>", tokens, re.IGNORECASE)
        if start_token is None:
            break
        key = start_token.group(1)
        end_token = re.search(rf"</s_{key}>", tokens, re.IGNORECASE)
        start_token = start_token.group()
        if end_token is None:
            tokens = tokens.replace(start_token, "")
        else:
            end_token = end_token.group()
            start_token_escaped = re.escape(start_token)
            end_token_escaped = re.escape(end_token)
            content = re.search(f"{start_token_escaped}(.*?){end_token_escaped}", tokens, re.IGNORECASE)
            if content is not None:
                content = content.group(1).strip()
                if r"<s_" in content and r"</s_" in content:  # non-leaf node
                    value = donut_token2json(tokenizer, content, is_inner_value=True)
                    if value:
                        if len(value) == 1:
                            value = value[0]
                        output[key] = value
                else:  # leaf nodes
                    output[key] = []
                    for leaf in content.split(r"<sep/>"):
                        leaf = leaf.strip()
                        if leaf in tokenizer.get_added_vocab() and leaf[0] == "<" and leaf[-2:] == "/>":
                            leaf = leaf[1:-2]  # for categorical special tokens
                        output[key].append(leaf)
                    if len(output[key]) == 1:
                        output[key] = output[key][0]

            tokens = tokens[tokens.find(end_token) + len(end_token) :].strip()
            if tokens[:6] == r"<sep/>":  # non-leaf nodes
                return [output] + donut_token2json(tokenizer, tokens[6:], is_inner_value=True)

    if len(output):
        return [output] if is_inner_value else output
    else:
        return [] if is_inner_value else {"text_sequence": tokens}


@add_end_docstrings(PIPELINE_INIT_ARGS)
class DocumentClassificationPipeline(ChunkPipeline):
    """
    Document classification pipeline using any `AutoModelForDocumentClassification`. This pipeline predicts the class of a
    document.

    This document classification pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"document-classification"`.

    See the list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=document-classification).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.model.config.__class__.__name__ == "VisionEncoderDecoderConfig":
            self.model_type = ModelType.VisionEncoderDecoder
        else:
            self.model_type = ModelType.Standard

    def _sanitize_parameters(
        self,
        doc_stride=None,
        lang: Optional[str] = None,
        tesseract_config: Optional[str] = None,
        max_num_spans: Optional[int] = None,
        max_seq_len=None,
        function_to_apply=None,
        top_k=None,
    ):
        preprocess_params, postprocess_params = {}, {}
        if doc_stride is not None:
            preprocess_params["doc_stride"] = doc_stride
        if max_seq_len is not None:
            preprocess_params["max_seq_len"] = max_seq_len
        if lang is not None:
            preprocess_params["lang"] = lang
        if tesseract_config is not None:
            preprocess_params["tesseract_config"] = tesseract_config
        if max_num_spans is not None:
            preprocess_params["max_num_spans"] = max_num_spans

        if isinstance(function_to_apply, str):
            function_to_apply = ClassificationFunction[function_to_apply.upper()]

        if function_to_apply is not None:
            postprocess_params["function_to_apply"] = function_to_apply

        if top_k is not None:
            if top_k < 1:
                raise ValueError(f"top_k parameter should be >= 1 (got {top_k})")
            postprocess_params["top_k"] = top_k

        return preprocess_params, {}, postprocess_params

    def __call__(self, image: Union[ImageOrName, List[ImageOrName], List[Tuple]], **kwargs):
        """
        Assign labels to the document(s) passed as inputs.

        # TODO
        """
        if isinstance(image, list):
            normalized_images = (i if isinstance(i, (tuple, list)) else (i, None) for i in image)
        else:
            normalized_images = [(image, None)]

        return super().__call__({"pages": normalized_images}, **kwargs)

    def preprocess(
        self,
        input,
        doc_stride=None,
        max_seq_len=None,
        word_boxes: Tuple[str, List[float]] = None,
        lang=None,
        tesseract_config="",
        max_num_spans=1,
    ):
        # NOTE: This code mirrors the code in question answering and will be implemented in a follow up PR
        # to support documents with enough tokens that overflow the model's window
        if max_seq_len is None:
            # TODO: LayoutLM's stride is 512 by default. Is it ok to use that as the min
            # instead of 384 (which the QA model uses)?
            max_seq_len = min(self.tokenizer.model_max_length, 512)

        if doc_stride is None:
            doc_stride = min(max_seq_len // 2, 256)

        total_num_spans = 0

        for page_idx, (image, word_boxes) in enumerate(input["pages"]):
            image_features = {}
            if image is not None:
                if not VISION_LOADED:
                    raise ValueError(
                        "If you provide an image, then the pipeline will run process it with PIL (Pillow), but"
                        " PIL is not available. Install it with pip install Pillow."
                    )
                image = load_image(image)
                if self.feature_extractor is not None:
                    image_features.update(self.feature_extractor(images=image, return_tensors=self.framework))

            words, boxes = None, None
            if self.model_type != ModelType.VisionEncoderDecoder:
                if word_boxes is not None:
                    words = [x[0] for x in word_boxes]
                    boxes = [x[1] for x in word_boxes]
                elif "words" in image_features and "boxes" in image_features:
                    words = image_features.pop("words")[0]
                    boxes = image_features.pop("boxes")[0]
                elif image is not None:
                    if not TESSERACT_LOADED:
                        raise ValueError(
                            "If you provide an image without word_boxes, then the pipeline will run OCR using"
                            " Tesseract, but pytesseract is not available. Install it with pip install pytesseract."
                        )
                    if TESSERACT_LOADED:
                        words, boxes = apply_tesseract(image, lang=lang, tesseract_config=tesseract_config)
                else:
                    raise ValueError(
                        "You must provide an image or word_boxes. If you provide an image, the pipeline will"
                        " automatically run OCR to derive words and boxes"
                    )

            if self.tokenizer.padding_side != "right":
                raise ValueError(
                    "Document classification only supports tokenizers whose padding side is 'right', not"
                    f" {self.tokenizer.padding_side}"
                )

            if self.model_type == ModelType.VisionEncoderDecoder:
                encoding = {
                    "inputs": image_features["pixel_values"],
                    "max_length": self.model.decoder.config.max_position_embeddings,
                    "decoder_input_ids": self.tokenizer(
                        "<s_rvlcdip>",
                        add_special_tokens=False,
                        return_tensors=self.framework,
                    ).input_ids,
                    "return_dict_in_generate": True,
                }
                yield {
                    **encoding,
                    "page": None,
                }
            else:
                encoding = self.tokenizer(
                    text=words,
                    max_length=max_seq_len,
                    stride=doc_stride,
                    return_token_type_ids=True,
                    is_split_into_words=True,
                    truncation=True,
                    return_overflowing_tokens=True,
                )

                num_spans = len(encoding["input_ids"])

                for span_idx in range(num_spans):
                    if self.framework == "pt":
                        span_encoding = {k: torch.tensor(v[span_idx : span_idx + 1]) for (k, v) in encoding.items()}
                        span_encoding.update(
                            {k: v for (k, v) in image_features.items()}
                        )  # TODO: Verify cardinality is correct
                    else:
                        raise ValueError("Unsupported: Tensorflow preprocessing for DocumentClassification")

                    # For each span, place a bounding box [0,0,0,0] for question and CLS tokens, [1000,1000,1000,1000]
                    # for SEP tokens, and the word's bounding box for words in the original document.
                    bbox = []
                    for i, s, w in zip(
                        encoding.input_ids[span_idx],
                        encoding.sequence_ids(span_idx),
                        encoding.word_ids(span_idx),
                    ):
                        if s == 0:
                            bbox.append(boxes[w])
                        elif i == self.tokenizer.sep_token_id:
                            bbox.append([1000] * 4)
                        else:
                            bbox.append([0] * 4)

                    span_encoding["bbox"] = torch.tensor(bbox).unsqueeze(0)

                    yield {
                        **span_encoding,
                        "page": page_idx,
                    }

                    total_num_spans += 1
                    if total_num_spans >= max_num_spans:
                        break

    def _forward(self, model_inputs):
        page = model_inputs.pop("page", None)

        if "overflow_to_sample_mapping" in model_inputs:
            model_inputs.pop("overflow_to_sample_mapping")

        if self.model_type == ModelType.VisionEncoderDecoder:
            model_outputs = self.model.generate(**model_inputs)
        else:
            model_outputs = self.model(**model_inputs)

        model_outputs["page"] = page
        model_outputs["attention_mask"] = model_inputs.get("attention_mask", None)
        return model_outputs

    def postprocess(self, model_outputs, function_to_apply=None, top_k=1, **kwargs):
        if function_to_apply is None:
            if self.model.config.num_labels == 1:
                function_to_apply = ClassificationFunction.SIGMOID
            elif self.model.config.num_labels > 1:
                function_to_apply = ClassificationFunction.SOFTMAX
            elif hasattr(self.model.config, "function_to_apply") and function_to_apply is None:
                function_to_apply = self.model.config.function_to_apply
            else:
                function_to_apply = ClassificationFunction.NONE

        if self.model_type == ModelType.VisionEncoderDecoder:
            answers = self.postprocess_encoder_decoder(model_outputs, top_k=top_k, **kwargs)
        else:
            answers = self.postprocess_standard(
                model_outputs, function_to_apply=function_to_apply, top_k=top_k, **kwargs
            )

        answers = sorted(answers, key=lambda x: x.get("score", 0), reverse=True)[:top_k]
        return answers

    def postprocess_encoder_decoder(self, model_outputs, **kwargs):
        classes = set()
        for model_output in model_outputs:
            for sequence in self.tokenizer.batch_decode(model_output.sequences):
                sequence = sequence.replace(self.tokenizer.eos_token, "").replace(self.tokenizer.pad_token, "")
                sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token
                classes.add(donut_token2json(self.tokenizer, sequence)["class"])

        # Return the first top_k unique classes we see
        return [{"label": v} for v in classes]

    def postprocess_standard(self, model_outputs, function_to_apply, **kwargs):
        # Average the score across pages
        sum_scores = {k: 0 for k in self.model.config.id2label.values()}
        for model_output in model_outputs:
            outputs = model_output["logits"][0]
            outputs = outputs.numpy()

            if function_to_apply == ClassificationFunction.SIGMOID:
                scores = sigmoid(outputs)
            elif function_to_apply == ClassificationFunction.SOFTMAX:
                scores = softmax(outputs)
            elif function_to_apply == ClassificationFunction.NONE:
                scores = outputs
            else:
                raise ValueError(f"Unrecognized `function_to_apply` argument: {function_to_apply}")

            for i, score in enumerate(scores):
                sum_scores[self.model.config.id2label[i]] += score.item()

            return [{"label": label, "score": score / len(model_outputs)} for (label, score) in sum_scores.items()]
