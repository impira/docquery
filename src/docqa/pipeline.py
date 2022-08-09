import logging
from transformers import AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from transformers.models.layoutlm.configuration_layoutlm import LayoutLMConfig
from transformers.pipelines import PIPELINE_REGISTRY

CHECKPOINT = "impira/layoutlm-document-qa"
REVISION = "3f1b351a6e4c32157b29097b88bbf5c111705a60"


def get_pipeline(checkpoint=None, revision=None):
    if checkpoint is None:
        checkpoint = CHECKPOINT

    if checkpoint == CHECKPOINT and revision is None:
        revision = REVISION

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        add_prefix_space=True,
        trust_remote_code=True,
        revision=revision,
    )

    return pipeline(
        model=checkpoint,
        tokenizer=tokenizer,
        trust_remote_code=True,
        revision=revision,
    )
