# This file is copied from transformers:
#    https://github.com/huggingface/transformers/blob/bb6f6d53386bf2340eead6a8f9320ce61add3e96/src/transformers/pipelines/image_classification.py
# And has been modified to support Donut
import re
from typing import List, Union

from transformers.pipelines.base import PIPELINE_INIT_ARGS, Pipeline
from transformers.utils import (
    ExplicitEnum,
    add_end_docstrings,
    is_tf_available,
    is_torch_available,
    is_vision_available,
    logging,
    requires_backends,
)


if is_vision_available():
    from PIL import Image
    from transformers.image_utils import load_image

if is_tf_available():
    import tensorflow as tf
    from transformers.models.auto.modeling_tf_auto import TF_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING
    from transformers.tf_utils import stable_softmax

if is_torch_available():
    from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING

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
class ImageClassificationPipeline(Pipeline):
    """
    Image classification pipeline using any `AutoModelForImageClassification`. This pipeline predicts the class of an
    image.

    This image classification pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"image-classification"`.

    See the list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=image-classification).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        requires_backends(self, "vision")
        self.check_model_type(
            TF_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING
            if self.framework == "tf"
            else MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING
        )

        if self.model.config.__class__.__name__ == "VisionEncoderDecoderConfig":
            self.model_type = ModelType.VisionEncoderDecoder
        else:
            self.model_type = ModelType.Standard

    def _sanitize_parameters(self, top_k=None):
        postprocess_params = {}
        if top_k is not None:
            postprocess_params["top_k"] = top_k
        return {}, {}, postprocess_params

    def __call__(self, images: Union[str, List[str], "Image.Image", List["Image.Image"]], **kwargs):
        """
        Assign labels to the image(s) passed as inputs.

        Args:
            images (`str`, `List[str]`, `PIL.Image` or `List[PIL.Image]`):
                The pipeline handles three types of images:

                - A string containing a http link pointing to an image
                - A string containing a local path to an image
                - An image loaded in PIL directly

                The pipeline accepts either a single image or a batch of images, which must then be passed as a string.
                Images in a batch must all be in the same format: all as http links, all as local paths, or all as PIL
                images.
            top_k (`int`, *optional*, defaults to 5):
                The number of top labels that will be returned by the pipeline. If the provided number is higher than
                the number of labels available in the model configuration, it will default to the number of labels.

        Return:
            A dictionary or a list of dictionaries containing result. If the input is a single image, will return a
            dictionary, if the input is a list of several images, will return a list of dictionaries corresponding to
            the images.

            The dictionaries contain the following keys:

            - **label** (`str`) -- The label identified by the model.
            - **score** (`int`) -- The score attributed by the model for that label.
        """
        return super().__call__(images, **kwargs)

    def preprocess(self, image):
        image = load_image(image)
        model_inputs = self.feature_extractor(images=image, return_tensors=self.framework)

        if self.model_type == ModelType.VisionEncoderDecoder:
            model_inputs = {
                "inputs": model_inputs["pixel_values"],
                "max_length": self.model.decoder.config.max_position_embeddings,
                "decoder_input_ids": self.tokenizer(
                    "<s_rvlcdip>",
                    add_special_tokens=False,
                    return_tensors=self.framework,
                ).input_ids,
                "return_dict_in_generate": True,
            }
        return model_inputs

    def _forward(self, model_inputs):
        if self.model_type == ModelType.VisionEncoderDecoder:
            model_outputs = self.model.generate(**model_inputs)
        else:
            model_outputs = self.model(**model_inputs)

        return model_outputs

    def postprocess(self, model_outputs, top_k=5):
        if self.model_type == ModelType.VisionEncoderDecoder:
            ret = []
            for sequence in self.tokenizer.batch_decode(model_outputs.sequences):
                sequence = sequence.replace(self.tokenizer.eos_token, "").replace(self.tokenizer.pad_token, "")
                sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token
                ret.append({"label": donut_token2json(self.tokenizer, sequence)["class"]})
            return ret[:top_k]

        if top_k > self.model.config.num_labels:
            top_k = self.model.config.num_labels

        if self.framework == "pt":
            probs = model_outputs.logits.softmax(-1)[0]
            scores, ids = probs.topk(top_k)
        elif self.framework == "tf":
            probs = stable_softmax(model_outputs.logits, axis=-1)[0]
            topk = tf.math.top_k(probs, k=top_k)
            scores, ids = topk.values.numpy(), topk.indices.numpy()
        else:
            raise ValueError(f"Unsupported framework: {self.framework}")

        scores = scores.tolist()
        ids = ids.tolist()
        return [{"score": score, "label": self.model.config.id2label[_id]} for score, _id in zip(scores, ids)]
