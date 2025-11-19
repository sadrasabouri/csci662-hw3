import pandas as pd

df = pd.read_csv('all_results.csv')

# Convert the metric columns to numeric, coercing empty strings/non-convertible values to NaN
metric_cols = [col for col in df.columns if '@' in col]
df[metric_cols] = df[metric_cols].apply(pd.to_numeric, errors='coerce')

# Identify the base metrics
base_metrics = ['avg-jenson', 'avg-precision', 'avg-recall', 'avg-f1']

# Initialize a new DataFrame with the identifier columns
new_df = df[['model_variant', 'k']].copy()

# Coalesce the values for each base metric
for metric in base_metrics:
    # Select columns that belong to this metric (e.g., 'avg-jenson@1', 'avg-jenson@10', ...)
    cols_to_coalesce = [col for col in metric_cols if col.startswith(f'{metric}@')]

    # Use bfill (backward fill) to select the single non-null value across the row.
    # The .iloc[:, 0] selects the resulting column containing the coalesced value.
    new_df[metric] = df[cols_to_coalesce].bfill(axis=1).iloc[:, 0]

new_df.to_csv('all_results_new.csv', index=False)
new_df.sort_values(by=['k', 'avg-recall'], ascending=[True, False]).to_csv('all_results_sorted_avg-recall.csv', index=False)
