import evaluate
import pandas as pd
from statistics import mean
from typing import Optional, Dict


def score_dataset(
    df: pd.DataFrame,
    *,
    metric: str,
    predictions_col: str,
    references_col: str,
    evaluate_args: Optional[Dict[str, str]] = None,
):
    results = evaluate.load(metric).compute(
        predictions=df[predictions_col],
        references=df[references_col],
        **(evaluate_args if evaluate_args else {}),
    )

    print(results.keys())

    print(f"Mean f1: {mean(results['f1'])}. over {len(df)} observations")


if __name__ == "__main__":
    df = pd.read_csv("data/candidates_references.csv")

    score_dataset(
        df,
        metric="bertscore",
        predictions_col="candidate",
        references_col="reference",
        evaluate_args={"lang": "en", "model_type": "distilbert-base-uncased"},
    )
