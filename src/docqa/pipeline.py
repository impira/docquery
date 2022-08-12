from transformers import AutoTokenizer, pipeline
from transformers.models.layoutlm.configuration_layoutlm import LayoutLMConfig
from transformers.pipelines import PIPELINE_REGISTRY

from .ext.model import LayoutLMForQuestionAnswering
from .ext.pipeline import DocumentQuestionAnsweringPipeline


CHECKPOINT = "impira/layoutlm-document-qa"
REVISION = "02daaaf614d4ae08fa6a1d51693baaf7de819585"


PIPELINE_REGISTRY.register_pipeline(
    "document-question-answering",
    pipeline_class=DocumentQuestionAnsweringPipeline,
    pt_model=LayoutLMForQuestionAnswering,
)


def get_pipeline(checkpoint=None, revision=None):
    if checkpoint is None:
        checkpoint = CHECKPOINT

    if checkpoint == CHECKPOINT and revision is None:
        revision = REVISION

    config = LayoutLMConfig.from_pretrained(checkpoint, revision=revision)

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        add_prefix_space=True,
        revision=revision,
        config=config,
    )

    model = LayoutLMForQuestionAnswering.from_pretrained(checkpoint, revision=revision)

    return pipeline(
        "document-question-answering",
        revision=revision,
        model=model,
        tokenizer=tokenizer,
    )
