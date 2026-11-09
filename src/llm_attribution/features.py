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

"""Feature extraction for stylometric text attribution."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion


def build_tfidf_union(
    word_ngrams=(1, 2),
    char_ngrams=(3, 5),
    min_df=3,
    max_df=0.9,
):
    """Word + character TF-IDF union - the workhorse representation.

    Character n-grams capture model-specific quirks (punctuation habits,
    whitespace, markup artifacts) that word features alone miss.
    """
    word = TfidfVectorizer(
        analyzer="word", ngram_range=word_ngrams, min_df=min_df,
        max_df=max_df, sublinear_tf=True,
    )
    char = TfidfVectorizer(
        analyzer="char_wb", ngram_range=char_ngrams, min_df=min_df,
        max_df=max_df, sublinear_tf=True,
    )
    return FeatureUnion([("word", word), ("char", char)])
