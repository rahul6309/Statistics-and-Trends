import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns
from corner import corner

def plot_relational_plot(df):
    """Plots relationship between Age and Household Size."""
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x='Respondent Age', y='household_size')
    plt.title('Age vs Household Size')
    plt.savefig('relational_plot.png')
    plt.close()

def plot_categorical_plot(df):
    """Plots Age distribution by Country."""
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x='country', y='Respondent Age')
    plt.xticks(rotation=45)
    plt.savefig('categorical_plot.png')
    plt.close()

def plot_statistical_plot(df):
    """Plots the distribution of Respondent Age."""
    plt.figure(figsize=(8, 5))
    sns.histplot(df['Respondent Age'], kde=True)
    plt.savefig('statistical_plot.png')
    plt.close()

def statistical_analysis(df, col: str):
    """Calculates mean, stddev, skew, and kurtosis."""
    data = df[col].dropna()  # Remove NaNs to prevent errors
    mean = data.mean()
    stddev = data.std()
    skew = ss.skew(data)
    excess_kurtosis = ss.kurtosis(data)
    return mean, stddev, skew, excess_kurtosis

def preprocessing(df):
    """Basic data cleaning and exploration."""
    print("Columns in dataset:", df.columns.tolist())
    # Convert numeric columns and drop missing values for analysis
    df['Respondent Age'] = pd.to_numeric(df['Respondent Age'], errors='coerce')
    df['household_size'] = pd.to_numeric(df['household_size'], errors='coerce')
    return df

def writing(moments, col):
    """Outputs the statistical results."""
    print(f'\nFor the attribute {col}:')
    print(f'Mean = {moments[0]:.2f}, '
          f'Standard Deviation = {moments[1]:.2f}, '
          f'Skewness = {moments[2]:.2f}, and '
          f'Excess Kurtosis = {moments[3]:.2f}.')
    
    # Interpretation
    skew_str = "right skewed" if moments[2] > 0 else "left skewed"
    kurt_str = "leptokurtic" if moments[3] > 0 else "platykurtic"
    print(f'The data was {skew_str} and {kurt_str}.')

def main():
    # Use the local filename
    try:
        df = pd.read_csv('Financial Dataset - 1.csv')
    except FileNotFoundError:
        print("Error: 'Financial Dataset - 1.csv' not found. Put the file in the same folder as this script.")
        return

    df = preprocessing(df)
    col = 'Respondent Age' # Analysis column
    
    plot_relational_plot(df)
    plot_statistical_plot(df)
    plot_categorical_plot(df)
    
    moments = statistical_analysis(df, col)
    writing(moments, col)

if __name__ == '__main__':
    main()
