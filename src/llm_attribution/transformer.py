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

"""Transformer fine-tuning track (DeBERTa-v3 via KerasHub, torch backend).

Runs anywhere with a GPU; designed to also work fully offline once the
preset directory is available locally.
"""

import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np


def build_classifier(preset_dir, num_classes, sequence_length=256):
    """Load a KerasHub DeBERTa-v3 text classifier from a local preset dir."""
    import keras
    import keras_hub

    pre = keras_hub.models.DebertaV3Preprocessor.from_preset(
        preset_dir, sequence_length=sequence_length
    )
    backbone = keras_hub.models.DebertaV3Backbone.from_preset(preset_dir)
    clf = keras_hub.models.DebertaV3Classifier(
        backbone=backbone, preprocessor=pre, num_classes=num_classes
    )
    clf.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.AdamW(learning_rate=2e-5, weight_decay=0.01),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    return clf


def fine_tune(clf, texts, labels, epochs=2, batch_size=8):
    y = np.asarray(labels, dtype="int32")
    clf.fit(list(texts), y, epochs=epochs, batch_size=batch_size, verbose=1)
    return clf


def predict_labels(clf, texts, classes, batch_size=16):
    logits = clf.predict(list(texts), batch_size=batch_size, verbose=1)
    return [classes[int(i)] for i in np.argmax(logits, axis=1)]
