"""
This is the template file for the statistics and trends assignment.
You will be expected to complete all the sections and
make this a fully working, documented file.
You should NOT change any function, file or variable names,
 if they are given to you here.
Make use of the functions presented in the lectures
and ensure your code is PEP-8 compliant, including docstrings.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss

def plot_relational_plot(df):
    """
    Plot scatter diagram with trend analysis showing power consumed
    versus charging time relationship.

    Parameters
    ----------
    df : pd.DataFrame
        EV charging station data with power and time metrics.

    Returns
    -------
    None
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # Create scatter plot colored by charging type
    for charging_type in df['charging_type'].unique():
        subset = df[df['charging_type'] == charging_type]
        ax.scatter(subset['charging_time'], subset['power_consumed'],
                   label=charging_type, alpha=0.6, s=40, edgecolors='w',
                   linewidth=0.5)

    ax.set_xlabel('Charging Time (minutes)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Power Consumed (kWh)', fontsize=13, fontweight='bold')
    ax.set_title('EV Charging: Power Consumption vs Charging Duration',
                 fontsize=15, fontweight='bold', pad=18)
    ax.legend(title='Charging Type', fontsize=10, title_fontsize=11)
    ax.grid(True, alpha=0.25, linestyle='-')

    plt.tight_layout()
    plt.savefig('relational_plot.png', dpi=300)
    plt.close()
    return


def plot_categorical_plot(df):
    """
    Create stacked bar chart illustrating the distribution of
    reduced power loss categories across different locations.

    Parameters
    ----------
    df : pd.DataFrame
        EV charging station data with categorical features.

    Returns
    -------
    None
    """
    fig, ax = plt.subplots(figsize=(11, 7))

    # Create cross-tabulation
    ct = pd.crosstab(df['location'], df['reduced_power_loss_category'])

    # Plot stacked bars
    ct.plot(kind='bar', stacked=True, ax=ax,
            color=['#E63946', '#F1FAEE', '#A8DADC'],
            edgecolor='black', linewidth=1.2)

    ax.set_xlabel('Location Type', fontsize=13, fontweight='bold')
    ax.set_ylabel('Count', fontsize=13, fontweight='bold')
    ax.set_title('Power Loss Categories by Location',
                 fontsize=15, fontweight='bold', pad=18)
    ax.legend(title='Power Loss Category', bbox_to_anchor=(1.05, 1),
              loc='upper left', fontsize=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig('categorical_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    return


def plot_statistical_plot(df):
    """
    Construct multiple box plots comparing grid stability scores
    across voltage stability categories.

    Parameters
    ----------
    df : pd.DataFrame
        EV charging station data with stability metrics.

    Returns
    -------
    None
    """
    fig, ax = plt.subplots(figsize=(11, 7))

    # Prepare data for box plot
    stability_order = ['Poor', 'Moderate', 'Excellent']
    available_categories = [cat for cat in stability_order
                            if cat in df['voltage_stability_category'].values]

    box_data = [
     df[df['voltage_stability_category'] == cat][
     'grid_stability_score'
    ].values
    for cat in available_categories
    ]

    # Create box plot with custom styling
    bplot = ax.boxplot(box_data, labels=available_categories,
                       patch_artist=True, notch=False,
                       widths=0.6,
                       medianprops=dict(color='red', linewidth=2.5),
                       boxprops=dict(linewidth=1.5),
                       whiskerprops=dict(linewidth=1.5),
                       capprops=dict(linewidth=1.5))

    # Color boxes with gradient
    color_palette = ['#FFA07A', '#87CEEB', '#98FB98']
    for patch, color in zip(bplot['boxes'], color_palette[:len(box_data)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel('Voltage Stability Category', fontsize=13, fontweight='bold')
    ax.set_ylabel('Grid Stability Score', fontsize=13, fontweight='bold')
    ax.set_title('Grid Stability Analysis by Voltage Category',
                 fontsize=15, fontweight='bold', pad=18)
    ax.grid(axis='y', alpha=0.3, linestyle=':')

    plt.tight_layout()
    plt.savefig('statistical_plot.png', dpi=300)
    plt.close()
    return


def statistical_analysis(df, col: str):
    """
    Calculate four fundamental moments describing the shape
    and characteristics of the data distribution.

    Parameters
    ----------
    df : pd.DataFrame
        EV charging dataset.
    col : str
        Column name for statistical moment calculation.

    Returns
    -------
    tuple
        Four moments: (mean, stddev, skew, excess_kurtosis).
    """
    data_values = df[col].values
    mean = np.mean(data_values)
    stddev = np.std(data_values, ddof=1)
    skew = ss.skew(data_values, nan_policy='omit', bias=True)
    excess_kurtosis = ss.kurtosis(data_values, nan_policy='omit', bias=True)
    return mean, stddev, skew, excess_kurtosis


def preprocessing(df):
    """
    Perform data preprocessing including exploratory analysis
    with describe, head, tail, and correlation methods.

    Parameters
    ----------
    df : pd.DataFrame
        Raw EV charging station dataset.

    Returns
    -------
    pd.DataFrame
        Cleaned and preprocessed dataset.
    """
    # You should preprocess your data in this function and
    # make use of quick features such as 'describe', 'head/tail' and 'corr'.

    print("\n" + "+" * 85)
    print("+{:^83}+".format("DATA PREPROCESSING AND EXPLORATORY ANALYSIS"))
    print("+" * 85 + "\n")

    print("[1] DATASET DIMENSIONS")
    print(f"    Rows: {df.shape[0]:,} | Columns: {df.shape[1]}\n")

    print("[2] FIRST 5 RECORDS (HEAD)")
    head_data = df.head(5)
    print(head_data.to_string(max_colwidth=15))
    print()

    print("[3] LAST 5 RECORDS (TAIL)")
    tail_data = df.tail(5)
    print(tail_data.to_string(max_colwidth=15))
    print()

    print("[4] STATISTICAL DESCRIPTION")
    description = df.describe()
    print(description.to_string())
    print()

    print("[5] CORRELATION MATRIX (NUMERICAL FEATURES)")
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    # Display subset of correlation matrix for readability
    print(corr_matrix.iloc[:6, :6].to_string())
    print("    ... (truncated for brevity)")
    print()

    print("[6] MISSING VALUES CHECK")
    missing_vals = df.isnull().sum()
    print(missing_vals[missing_vals > 0] if missing_vals.sum() > 0
          else "    No missing values detected!")
    print()

    print("[7] DATA TYPE INFORMATION")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"    {dtype}: {count} columns")
    print()

    # Cleaning pipeline
    rows_before = len(df)
    df_clean = df.dropna()
    df_clean = df_clean.drop_duplicates()
    rows_after = len(df_clean)

    print("[8] CLEANING OPERATIONS")
    print(f"    Before: {rows_before:,} rows")
    print(f"    After:  {rows_after:,} rows")
    print(f"    Removed: {rows_before - rows_after:,} rows")
    print()

    print("+" * 85 + "\n")

    return df_clean


def writing(moments, col):
    """
    Display computed statistical moments and interpret
    the distribution's shape characteristics.

    Parameters
    ----------
    moments : tuple
        Four moments (mean, stddev, skew, excess_kurtosis).
    col : str
        Name of analyzed column.

    Returns
    -------
    None
    """
    print(f'\nFor the attribute {col}:')
    print(f'Mean = {moments[0]:.2f}, '
          f'Standard Deviation = {moments[1]:.2f}, '
          f'Skewness = {moments[2]:.2f}, and '
          f'Excess Kurtosis = {moments[3]:.2f}.')

    # Interpret skewness
    skew_value = moments[2]
    if skew_value < -2:
        skew_label = "left"
    elif skew_value > 2:
        skew_label = "right"
    else:
        skew_label = "not"

    # Interpret kurtosis
    kurt_value = moments[3]
    if kurt_value < 0:
        kurt_label = "platy"
    elif kurt_value > 0:
        kurt_label = "lepto"
    else:
        kurt_label = "meso"

    print(f'The data was {skew_label} skewed and {kurt_label}kurtic.')
    return


def main():
    """
    Execute complete analysis workflow including data loading,
    preprocessing, visualization, and statistical analysis.

    Returns
    -------
    None
    """
    df = pd.read_csv('data.csv')
    df = preprocessing(df)

    col = 'power_consumed'

    plot_relational_plot(df)
    plot_statistical_plot(df)
    plot_categorical_plot(df)

    moments = statistical_analysis(df, col)
    writing(moments, col)

    return


if __name__ == '__main__':
    main()
