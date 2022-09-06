from collections import OrderedDict

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer, pipeline
from transformers.models.auto.auto_factory import _BaseAutoModelClass, _LazyAutoMapping
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
from transformers.pipelines import PIPELINE_REGISTRY

from .ext.config import LayoutLMConfig, LayoutLMDocQueryConfig
from .ext.model import LayoutLMDocQueryForQuestionAnswering, LayoutLMForQuestionAnswering
from .ext.pipeline import DocumentQuestionAnsweringPipeline


CHECKPOINT = "impira/layoutlm-document-qa"
REVISION = "3a93017fc2d200d68d0c2cc0fa62eb8d50314605"

MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES = OrderedDict(
    [
        ("layoutlm", "LayoutLMForQuestionAnswering"),
        ("layoutlm-docquery", "LayoutLMDocQueryForQuestionAnswering"),
        ("donut-swin", "DonutSwinModel"),
    ]
)

MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES
)


class AutoModelForDocumentQuestionAnswering(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING


AutoConfig.register("layoutlm-docquery", LayoutLMDocQueryConfig)
AutoModel.register(LayoutLMDocQueryConfig, LayoutLMDocQueryForQuestionAnswering)
AutoModelForDocumentQuestionAnswering.register(LayoutLMDocQueryConfig, LayoutLMDocQueryForQuestionAnswering)


PIPELINE_REGISTRY.register_pipeline(
    "document-question-answering",
    pipeline_class=DocumentQuestionAnsweringPipeline,
    pt_model=AutoModelForDocumentQuestionAnswering,
)


def get_pipeline(checkpoint=None, revision=None, device=None, **pipeline_kwargs):
    if checkpoint is None:
        checkpoint = CHECKPOINT

    if checkpoint == CHECKPOINT and revision is None:
        revision = REVISION

    if device is None:
        device = 0 if torch.cuda.is_available() else -1

    kwargs = {}
    if checkpoint == CHECKPOINT:
        config = LayoutLMConfig.from_pretrained(checkpoint, revision=revision)
    else:
        config = AutoConfig.from_pretrained(checkpoint, revision=revision)

    if config.model_type in ("layoutlm", "layoutlm-docquery"):
        kwargs["add_prefix_space"] = True

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        revision=revision,
        config=config,
        **kwargs,
    )

    if checkpoint == CHECKPOINT:
        model = LayoutLMForQuestionAnswering.from_pretrained(checkpoint, revision=revision)
    else:
        model = checkpoint

    if config.model_type == "vision-encoder-decoder":
        pipeline_kwargs["feature_extractor"] = model
    elif config.model_type == "layoutlm-docquery":
        pipeline_kwargs["max_answer_len"] = 1000  # Let the token classifier filter things out

    return pipeline(
        "document-question-answering",
        revision=revision,
        model=model,
        tokenizer=tokenizer,
        device=device,
        **pipeline_kwargs,
    )
