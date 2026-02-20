"""
This is the template file for the statistics and trends assignment.
You will be expected to complete all the sections and
make this a fully working, documented file.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns


def plot_relational_plot(df):
    """Generates a scatter plot of Age vs Household Size."""
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='Respondent Age', y='household_size',
                    hue='Has a Bank account')
    plt.title('Relationship: Age vs Household Size')
    plt.savefig('relational_plot.png')
    plt.close()
    return


def plot_categorical_plot(df):
    """Generates a boxplot of Age across different countries."""
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='country', y='Respondent Age')
    plt.title('Age Distribution by Country')
    plt.xticks(rotation=45)
    plt.savefig('categorical_plot.png')
    plt.close()
    return


def plot_statistical_plot(df):
    """Generates a histogram showing the distribution of Respondent Age."""
    plt.figure(figsize=(8, 6))
    sns.histplot(df['Respondent Age'], kde=True, color='blue')
    plt.title('Statistical Distribution of Respondent Age')
    plt.savefig('statistical_plot.png')
    plt.close()
    return


def statistical_analysis(df, col: str):
    """Calculates Mean, Std Dev, Skewness, and Excess Kurtosis."""
    clean_data = pd.to_numeric(df[col], errors='coerce').dropna()
    mean = clean_data.mean()
    stddev = clean_data.std()
    skew = ss.skew(clean_data)
    excess_kurtosis = ss.kurtosis(clean_data)
    return mean, stddev, skew, excess_kurtosis


def preprocessing(df):
    """Cleans data and prints initial exploration metrics."""
    print(df.head())
    print(df.describe())
    # Correlation only works on numeric data
    print(df.select_dtypes(include=[np.number]).corr())
    df = df.dropna(subset=['Respondent Age', 'household_size', 'country'])
    return df


def writing(moments, col):
    """Prints the statistical findings and interpretations."""
    print(f'For the attribute {col}:')
    print(f'Mean = {moments[0]:.2f}, '
          f'Standard Deviation = {moments[1]:.2f}, '
          f'Skewness = {moments[2]:.2f}, and '
          f'Excess Kurtosis = {moments[3]:.2f}.')

    s = moments[2]
    k = moments[3]

    skew_res = "not skewed"
    if s > 0.5:
        skew_res = "right skewed"
    elif s < -0.5:
        skew_res = "left skewed"

    kurt_res = "mesokurtic"
    if k > 0.5:
        kurt_res = "leptokurtic"
    elif k < -0.5:
        kurt_res = "platykurtic"

    print(f'The data was {skew_res} and {kurt_res}.')
    return


def main():
    """Main function to run the analysis pipeline."""
    # Ensure the file is in the same directory
    df = pd.read_csv('data.csv')
    df = preprocessing(df)
    col = 'Respondent Age'
    plot_relational_plot(df)
    plot_statistical_plot(df)
    plot_categorical_plot(df)
    moments = statistical_analysis(df, col)
    writing(moments, col)
    return


if __name__ == '__main__':
    main()
