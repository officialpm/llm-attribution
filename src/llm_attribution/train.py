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

"""Command-line training and evaluation entry point.

Example:
    python -m llm_attribution.train --data corpus.csv         --text-col text --label-col model --group-col prompt_id
"""

import argparse

import pandas as pd

from .evaluate import grouped_cv_score
from .linear import build_linear_baseline


def main():
    p = argparse.ArgumentParser(description="LLM text attribution trainer")
    p.add_argument("--data", required=True, help="CSV with one row per passage")
    p.add_argument("--text-col", default="text")
    p.add_argument("--label-col", default="model")
    p.add_argument("--group-col", default=None,
                   help="column whose values must not cross CV folds")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--C", type=float, default=4.0)
    args = p.parse_args()

    df = pd.read_csv(args.data)
    texts = df[args.text_col].tolist()
    labels = df[args.label_col].to_numpy()
    groups = (df[args.group_col].to_numpy() if args.group_col
              else df.index.to_numpy())

    mean, std, folds = grouped_cv_score(
        lambda: build_linear_baseline(C=args.C), texts, labels, groups,
        n_splits=args.folds,
    )
    print("per-fold balanced accuracy:", ", ".join(f"{s:.4f}" for s in folds))
    print(f"grouped CV balanced accuracy: {mean:.4f} +- {std:.4f}")


if __name__ == "__main__":
    main()
