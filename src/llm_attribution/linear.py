# Copyright 2026 Parth Maniar
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

"""Linear TF-IDF baseline for LLM text attribution."""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .features import build_tfidf_union


def build_linear_baseline(C=4.0, max_iter=2000):
    """TF-IDF (word 1-2 + char_wb 3-5) -> multinomial logistic regression."""
    return Pipeline([
        ("features", build_tfidf_union()),
        ("clf", LogisticRegression(C=C, max_iter=max_iter, n_jobs=-1)),
    ])
