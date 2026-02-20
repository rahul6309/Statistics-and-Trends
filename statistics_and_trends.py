"""
This is the completed assignment for statistics and trends.
The script processes financial data, calculates statistical moments,
and generates three distinct plots.
"""

from corner import corner
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns


def plot_relational_plot(df):
    """Generates a scatter plot of Age vs Household Size."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df, x='Respondent Age', y='household_size', 
                    hue='Has a Bank account', ax=ax)
    ax.set_title('Relationship: Age vs Household Size')
    plt.savefig('relational_plot.png')
    plt.close()
    return


def plot_categorical_plot(df):
    """Generates a boxplot of Age across different countries."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x='country', y='Respondent Age', ax=ax)
    ax.set_title('Age Distribution by Country')
    plt.xticks(rotation=45)
    plt.savefig('categorical_plot.png')
    plt.close()
    return


def plot_statistical_plot(df):
    """Generates a histogram showing the distribution of Respondent Age."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(df['Respondent Age'], kde=True, color='blue', ax=ax)
    ax.set_title('Statistical Distribution of Respondent Age')
    plt.savefig('statistical_plot.png')
    plt.close()
    return


def statistical_analysis(df, col: str):
    """Calculates Mean, Std Dev, Skewness, and Excess Kurtosis."""
    # Ensure we use numeric data and drop missing values for calculations
    clean_data = pd.to_numeric(df[col], errors='coerce').dropna()
    
    mean = clean_data.mean()
    stddev = clean_data.std()
    skew = ss.skew(clean_data)
    excess_kurtosis = ss.kurtosis(clean_data)
    
    return mean, stddev, skew, excess_kurtosis


def preprocessing(df):
    """Cleans data and prints initial exploration metrics."""
    # Using describe, head, and corr as requested in the template
    print("--- Dataset Head ---")
    print(df.head())
    print("\n--- Summary Statistics ---")
    print(df.describe())
    print("\n--- Correlation Matrix ---")
    # Correlation only works on numeric data
    print(df.select_dtypes(include=[np.number]).corr())
    
    # Cleaning: Standardizing column names and removing NaNs for core analysis
    df = df.dropna(subset=['Respondent Age', 'household_size', 'country'])
    return df


def writing(moments, col):
    """Prints the statistical findings and interpretations."""
    print(f'\nFor the attribute {col}:')
    print(f'Mean = {moments[0]:.2f}, '
          f'Standard Deviation = {moments[1]:.2f}, '
          f'Skewness = {moments[2]:.2f}, and '
          f'Excess Kurtosis = {moments[3]:.2f}.')
    
    # Interpretation based on calculated moments
    # We define 'not skewed' and 'mesokurtic' within a +/- 0.5 range
    s = moments[2]
    k = moments[3]
    
    skew_res = "not skewed" if -0.5 < s < 0.5 else ("right skewed" if s > 0.5 else "left skewed")
    kurt_res = "mesokurtic" if -0.5 < k < 0.5 else ("leptokurtic" if k > 0.5 else "platykurtic")
    
    print(f'The data was {skew_res} and {kurt_res}.')
    return


def main():
    # Make sure 'Financial Dataset - 1.csv' is in the same folder as this script
    try:
        df = pd.read_csv('Financial Dataset - 1.csv')
    except FileNotFoundError:
        print("Error: The file 'Financial Dataset - 1.csv' was not found.")
        print("Please ensure the CSV file is in the same folder as this script.")
        return

    df = preprocessing(df)
    col = 'Respondent Age' # Analyzing the Age column
    
    plot_relational_plot(df)
    plot_statistical_plot(df)
    plot_categorical_plot(df)
    
    moments = statistical_analysis(df, col)
    writing(moments, col)
    return


if __name__ == '__main__':
    main()
