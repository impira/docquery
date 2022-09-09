# This file contains extensions to transformers that have not yet been upstreamed. Importantly, since docquery
# is designed to be easy to install via PyPI, we must extend anything that is not part of an official release,
# since libraries on pypi are not permitted to install specific git commits.

from collections import OrderedDict
from typing import Optional, Union

import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForImageClassification,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    VisionEncoderDecoderConfig,
    VisionEncoderDecoderModel,
)
from transformers import pipeline as transformers_pipeline
from transformers.models.auto.auto_factory import _BaseAutoModelClass, _LazyAutoMapping
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
from transformers.pipelines import PIPELINE_REGISTRY

from .ext.config import LayoutLMConfig, LayoutLMTokenClassifierConfig
from .ext.model import LayoutLMForQuestionAnswering, LayoutLMTokenClassifierForQuestionAnswering
from .ext.pipeline_document_question_answering import DocumentQuestionAnsweringPipeline
from .ext.pipeline_image_classification import ImageClassificationPipeline


PIPELINE_DEFAULTS = {
    "document-question-answering": "impira/layoutlm-document-qa",
    "image-classification": "naver-clova-ix/donut-base-finetuned-rvlcdip",
}

# These revisions are pinned so that the "default" experience in DocQuery is both fast (does not
# need to check network for updates) and versioned (we can be sure that the model changes
# result in new versions of DocQuery). This may eventually change.
DEFAULT_REVISIONS = {
    "impira/layoutlm-document-qa": "ff904df",
    "impira/layoutlm-invoices": "f3db68b",
    "naver-clova-ix/donut-base-finetuned-rvlcdip": "5998e9f",
}

MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES = OrderedDict(
    [
        ("layoutlm", "LayoutLMForQuestionAnswering"),
        ("layoutlm-tc", "LayoutLMTokenClassifierForQuestionAnswering"),
        ("donut-swin", "DonutSwinModel"),
    ]
)

MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES
)


class AutoModelForDocumentQuestionAnswering(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING


AutoConfig.register("layoutlm-tc", LayoutLMTokenClassifierConfig)
AutoModel.register(LayoutLMTokenClassifierConfig, LayoutLMTokenClassifierForQuestionAnswering)
AutoModelForDocumentQuestionAnswering.register(
    LayoutLMTokenClassifierConfig, LayoutLMTokenClassifierForQuestionAnswering
)


PIPELINE_REGISTRY.register_pipeline(
    "document-question-answering",
    pipeline_class=DocumentQuestionAnsweringPipeline,
    pt_model=AutoModelForDocumentQuestionAnswering,
)

# The Donut model used for image classification defined here: https://huggingface.co/naver-clova-ix/donut-base-finetuned-rvlcdip
# declares itself a VisionEncoderDecoderModel, not a DonutModel, so we have to register the former in the Auto Class. We may want
# to fork this model and setup a different configuration that specifies Donut
AutoModelForImageClassification.register(VisionEncoderDecoderConfig, VisionEncoderDecoderModel)

PIPELINE_REGISTRY.register_pipeline(
    "image-classification",
    pipeline_class=ImageClassificationPipeline,
    pt_model=AutoModelForImageClassification,
)


def pipeline(
    task: str = None,
    model: Optional = None,
    tokenizer: Optional[Union[str, PreTrainedTokenizer, PreTrainedTokenizerFast]] = None,
    revision: Optional[str] = None,
    device: Optional[Union[int, str, "torch.device"]] = None,
    **pipeline_kwargs
):

    if model is None and task is not None:
        model = PIPELINE_DEFAULTS.get(task)

    if revision is None and model is not None:
        revision = DEFAULT_REVISIONS.get(model)

    # We need to explicitly check for the impira/layoutlm-document-qa model because of challenges with
    # registering an existing model "flavor" (layoutlm) within transformers after the fact. There may
    # be a clever way to get around this. Either way, we should be able to remove it once
    # https://github.com/huggingface/transformers/commit/5c4c869014f5839d04c1fd28133045df0c91fd84
    # is officially released.
    if model == "impira/layoutlm-document-qa":
        config = LayoutLMConfig.from_pretrained(model, revision=revision)
    else:
        config = AutoConfig.from_pretrained(model, revision=revision)

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            model,
            revision=revision,
            config=config,
        )

    if model == "impira/layoutlm-document-qa":
        model = LayoutLMForQuestionAnswering.from_pretrained(model, revision=revision)

    if config.model_type == "vision-encoder-decoder":
        # This _should_ be a feature of transformers -- deriving the feature_extractor automatically --
        # but is not at the time of writing, so we do it explicitly.
        pipeline_kwargs["feature_extractor"] = model

    if device is None:
        # This trick merely simplifies the device argument, so that cuda is used by default if it's
        # available, which at the time of writing is not a feature of transformers
        device = 0 if torch.cuda.is_available() else -1

    return transformers_pipeline(
        task,
        revision=revision,
        model=model,
        tokenizer=tokenizer,
        device=device,
        **pipeline_kwargs,
    )
