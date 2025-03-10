import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import numpy as np

def show_missing_values(df):
    def min_or_nan(col):
        if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
            return str(df[col].min().round(2))
        return np.nan
    def max_or_nan(col):
        if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
            return str(df[col].max().round(2))
        return np.nan

    missing_df = pd.DataFrame({
        'S. No.': range(1, len(df.columns) + 1),
        'Column Name': df.columns,
        'Min': [min_or_nan(col) for col in df.columns],
        'Max': [max_or_nan(col) for col in df.columns],
        'n Unique': df.nunique(),
        'NaN count': df.isna().sum(),
        'NaN percentage': (df.isna().mean() * 100).round(3).astype(str) + '%',
        'dtype': df.dtypes.astype(str),

    }).set_index('S. No.')

    unique_dtypes = missing_df['dtype'].unique()
    palette = sns.color_palette("Set2", n_colors=len(unique_dtypes))
    dtype_color_map = {dt: f"background-color: {mcolors.to_hex(color)}" for dt, color in zip(unique_dtypes, palette)}

    def color_row(row):
        return [dtype_color_map.get(row['dtype'], "")] * len(row)

    return missing_df.style.apply(color_row, axis=1)

def impute_mmm(series):
    series = series.dropna()
    mean = series.mean()
    median = series.median()
    mode = series.mode().values[0]
    count = series.count()
    mean_count = (series == mean).sum()
    median_count = (series == median).sum()
    mode_count = (series == mode).sum()
    mean_perc = (mean_count / count) * 100
    median_perc = (median_count / count) * 100
    mode_perc = (mode_count / count) * 100
    lower_bound = series.quantile(0.25)
    upper_bound = series.quantile(0.75)
    iqr = upper_bound - lower_bound
    lower_outliers_mask = series < (lower_bound - 1.5 * iqr)
    upper_outliers_mask = series > (upper_bound + 1.5 * iqr)
    outliers_mask = lower_outliers_mask | upper_outliers_mask
    series_without_outliers = series[~outliers_mask]
    mean_without_outliers = series_without_outliers.mean()
    median_without_outliers = series_without_outliers.median()
    mode_without_outliers = series_without_outliers.mode().values[0]
    count_without_outliers = series_without_outliers.count()
    mean_count_without_outliers = (series_without_outliers == mean_without_outliers).sum()
    median_count_without_outliers = (series_without_outliers == median_without_outliers).sum()
    mode_count_without_outliers = (series_without_outliers == mode_without_outliers).sum()
    mean_perc_without_outliers = (mean_count_without_outliers / count_without_outliers) * 100
    median_perc_without_outliers = (median_count_without_outliers / count_without_outliers) * 100
    mode_perc_without_outliers = (mode_count_without_outliers / count_without_outliers) * 100
    return pd.DataFrame({
        'Statistics': ['Mean', 'Median', 'Mode', 'Mean without Outliers', 'Median without Outliers', 'Mode without Outliers'],
        'Value': [mean, median, mode, mean_without_outliers, median_without_outliers, mode_without_outliers],
        'Count': [mean_count, median_count, mode_count, mean_count_without_outliers, median_count_without_outliers, mode_count_without_outliers],
        'Percentage': [mean_perc, median_perc, mode_perc, mean_perc_without_outliers, median_perc_without_outliers, mode_perc_without_outliers]
    }).set_index('Statistics')

def plot_top_categories(series, n=10, ax=None, title=None):
    if ax is None:
        ax = plt.gca()
    top_categories = series.value_counts().head(n).index
    sns.countplot(x=series, order=top_categories, ax=ax)
    if title:
        ax.set_title(title)
    ax.set_ylabel('Frequency')
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{int(height)}',
                    (p.get_x() + p.get_width() / 2, height),
                    ha='center', va='bottom',
                    xytext=(0, 3),
                    textcoords='offset points')
    ax.tick_params(axis='y', which='both', left=False, labelleft=False)

def plot_histogram(series, gap=5, ax=None, title=None):
    if ax is None:
        ax = plt.gca()
    sns.histplot(series, bins=int(series.max() - series.min()), kde=True, ax=ax)
    if title:
        ax.set_title(title)
    mean = series.mean()
    median = series.median()
    ax.axvline(mean, color='red', linestyle='dashed', linewidth=2, label=f'{mean = :.2f}')
    ax.axvline(median, color='green', linestyle='dashed', linewidth=2, label=f'{median = :.2f}')
    ax.set_xticks(range(int(series.min()), int(series.max()), gap))
    ax.legend()

def show_outliers(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    return outliers

def show_mar_relation(df, target_col, top_n=10, gap=5):
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not in DataFrame")
    if df[target_col].isna().sum() == 0:
        return f'No missing values in {target_col}'
    not_missing = df[df[target_col].notnull()]
    missing = df[df[target_col].isnull()]
    for col in df.columns:
        if col == target_col:
            continue 
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
            plot_func = plot_histogram
        else:
            plot_func = plot_top_categories 
        plot_func(not_missing[col].dropna(), ax=axes[0])
        axes[0].set_title(f"{col} (target not missing)")
        plot_func(missing[col].dropna(), ax=axes[1])
        axes[1].set_title(f"{col} (target missing)")
        plt.tight_layout()

def impute_random(series, random_seed=None):
    imputed_series = series.copy()
    missing_mask = imputed_series.isna()
    n_missing = missing_mask.sum()
    if n_missing == 0:
        return imputed_series
    imputed_values = series.dropna().sample(n_missing, replace=True, random_state=random_seed)
    imputed_series[missing_mask] = imputed_values.values
    return imputed_series

def convert_to_categorical(df, columns):
    df_ = df.copy()
    for col in columns:
        assert col in df.columns, f"Column {col} is not present in df"
        df_[col] = df_[col].astype('object')
    return df_