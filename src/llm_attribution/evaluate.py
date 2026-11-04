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

"""Evaluation utilities: grouped cross-validation and balanced accuracy."""

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold


def grouped_cv_score(model_fn, texts, labels, groups, n_splits=5, seed=42):
    """GroupKFold CV so no prompt/context group leaks across folds.

    model_fn: zero-arg callable returning a fresh fit/predict estimator.
    Returns (mean, std, per_fold) of balanced accuracy.
    """
    gkf = GroupKFold(n_splits=n_splits)
    scores = []
    for tr, va in gkf.split(texts, labels, groups):
        clf = model_fn()
        clf.fit([texts[i] for i in tr], labels[tr])
        pred = clf.predict([texts[i] for i in va])
        scores.append(balanced_accuracy_score(labels[va], pred))
    scores = np.asarray(scores)
    return scores.mean(), scores.std(), scores
