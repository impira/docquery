# coding=utf-8
# Copyright 2010, The Microsoft Research Asia LayoutLM Team authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" LayoutLM model configuration"""
from transformers import LayoutLMConfig


class LayoutLMDocQueryConfig(LayoutLMConfig):
    model_type = "layoutlm-docquery"

    def __init__(
        # New stuff added for DocQuery
        token_classification=False,
        token_classifier_reduction="mean",
        token_classifier_constant=1.0,
        **kwargs
    ):
        super().__init__(
            token_classification=token_classification,
            token_classifier_reduction=token_classifier_reduction,
            token_classifier_constant=token_classifier_constant,
            **kwargs,
        )
