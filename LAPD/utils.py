import pandas as pd
def check_missing_values(df):
    return pd.DataFrame({
        'Missing_Count': df.isna().sum(),
        'Missing_Percentage': (df.isna().sum().round(2) / len(df)) * 100
    })

def show_num_ranges(df):
    df_num = df.select_dtypes(include=['number'])
    return pd.DataFrame({
        'Min': df_num.min().round(2),
        'Max': df_num.max().round(2),
        'dtype': df_num.dtypes.astype(str),
        'NaN Count': df_num.isna().sum()
    })

def show_cat_values(df):
    df_cat = df.select_dtypes(include=['object', 'category', 'bool', 'boolean'])
    return pd.DataFrame({
        'Unique Values': df_cat.nunique(),
        'dtype': df_cat.dtypes.astype(str),
        'NaN Count': df_cat.isna().sum()
    })

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