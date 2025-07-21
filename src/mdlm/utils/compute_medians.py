import pandas as pd
from common.utils.column_selector import select_numeric_columns


def _compute_bottom_top_medians(df: pd.DataFrame, exclude: set, output_csv: str):
    cols_to_process = select_numeric_columns(df, exclude)
    results = {}
    for col in cols_to_process:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(series) == 0:
            results[col] = {"bottom33_median": None, "top33_median": None}
            continue
        lower_third = series.quantile(0.33)
        upper_third = series.quantile(0.67)
        bottom_partition = series[series <= lower_third]
        top_partition = series[series >= upper_third]
        bottom_median = bottom_partition.median() if len(bottom_partition) > 0 else None
        top_median = top_partition.median() if len(top_partition) > 0 else None
        results[col] = {"bottom33_median": bottom_median, "top33_median": top_median}
    result_df = pd.DataFrame(results).T
    result_df.to_csv(output_csv, index=True)
    print(f"Saved bottom/top 33rd percentile medians to {output_csv}")
    return result_df


if __name__ == "__main__":
    df = pd.read_csv(
        "/scratch/s3905845/thesis_final_coding/data/kraken/training_data/filtered_selfies.csv"
    )
    exclude = {"smiles", "selfies", "tokenized_selfies"}
    output_csv = "/scratch/s3905845/thesis_final_coding/data/kraken/training_data/bottom_top_medians.csv"
    medians_df = _compute_bottom_top_medians(df, exclude, output_csv)
