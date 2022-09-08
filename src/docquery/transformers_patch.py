# This file contains extensions to transformers that have not yet been upstreamed. Importantly, since docquery
# is designed to be easy to install via PyPI, we must extend anything that is not part of an official release,
# since libraries on pypi are not permitted to install specific git commits.

from collections import OrderedDict

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers import pipeline as transformers_pipeline
from transformers.models.auto.auto_factory import _BaseAutoModelClass, _LazyAutoMapping
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
from transformers.pipelines import PIPELINE_REGISTRY

from .config import get_logger
from .ext.config import LayoutLMConfig, LayoutLMTokenClassifierConfig
from .ext.model import LayoutLMForQuestionAnswering, LayoutLMTokenClassifierForQuestionAnswering
from .ext.pipeline import DocumentQuestionAnsweringPipeline


log = get_logger("pipeline")

CHECKPOINT = "impira/layoutlm-document-qa"
REVISION = "ff904df"

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


def pipeline(task=None, model=None, revision=None, device=None, tokenizer=None, **pipeline_kwargs):
    if task == "document-question-answering":
        if model is None:
            model = CHECKPOINT

        if model == CHECKPOINT and revision is None:
            # This revision is pinned so that the "default" experience in DocQuery is both fast (does not
            # need to check network for updates) and versioned (we can be sure that the model changes
            # result in new versions of DocQuery). This may eventually change.
            revision = REVISION

        # We need to explicitly check for the impira/layoutlm-document-qa model because of challenges with
        # registering an existing model "flavor" (layoutlm) within transformers after the fact. There may
        # be a clever way to get around this. Either way, we should be able to remove it once
        # https://github.com/huggingface/transformers/commit/5c4c869014f5839d04c1fd28133045df0c91fd84
        # is officially released.
        if model == CHECKPOINT:
            config = LayoutLMConfig.from_pretrained(model, revision=revision)
        else:
            config = AutoConfig.from_pretrained(model, revision=revision)

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                model,
                revision=revision,
                config=config,
            )

        if model == CHECKPOINT:
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
