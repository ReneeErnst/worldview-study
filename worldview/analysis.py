import pandas as pd
import pingouin as pg
from statsmodels.formula.api import ols
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns


def correlation_matrix(df_corrs: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a cleaned-up correlation matrix with additional statistics.

    This function computes pairwise correlations using Pingouin's `pairwise_corr`
    function and then enhances the resulting DataFrame by adding degrees of freedom,
    renaming columns for clarity, and removing unnecessary columns.

    Args:
        df_corrs (pd.DataFrame): DataFrame containing the data for correlation analysis.

    Returns:
        pd.DataFrame: A cleaned-up DataFrame containing correlation coefficients,
                      confidence intervals, p-values, degrees of freedom, and sample sizes.
    """

    df_corr_matrix = pg.pairwise_corr(df_corrs)

    # Calculate and add the 'df' (degrees of freedom) column
    df_corr_matrix["df"] = df_corr_matrix["n"] - 2

    # Rename columns for clarity
    df_corr_matrix = df_corr_matrix.rename(
        columns={"n": "Sample Size", "p-unc": "p-value"}
    )

    # Columns to keep
    final_columns = ["X", "Y", "r", "CI95%", "p-value", "df", "Sample Size"]

    return df_corr_matrix[final_columns]


def check_anova_assumptions(df: pd.DataFrame, group_col: str, dv_col: str):
    """
    Checks the normality and homogeneity of variance assumptions for an ANOVA.

    Args:
        df (pd.DataFrame): The pandas DataFrame containing the data.
        group_col (str): The name of the column containing the group labels.
        dv_col (str): The name of the column containing the values for the dependent variable.
    """
    print("--- ANOVA Assumption Check ---")

    # --- 1. Normality of Residuals ---
    print("\n1. Normality of Residuals Test (Shapiro-Wilk)")
    # Fit the model to get residuals
    model = ols(f"{dv_col} ~ C({group_col})", data=df).fit()
    residuals = model.resid

    shapiro_stat, shapiro_p = stats.shapiro(residuals)
    print(f"   Statistic: {shapiro_stat:.3f}, p-value: {shapiro_p:.3f}")

    if shapiro_p > 0.05:
        print("   Conclusion: Residuals appear to be normally distributed (p > 0.05).")
    else:
        print(
            "   Conclusion: Residuals do not appear to be normally distributed (p <= 0.05)."
        )

    # Q-Q Plot for visual inspection
    sm.qqplot(residuals, line="45")
    plt.title("Q-Q Plot of ANOVA Residuals")
    plt.show()

    # --- 2. Homogeneity of Variances ---
    print("\n2. Homogeneity of Variances Test (Levene's)")

    # Get the data for each group
    groups = df[group_col].unique()
    grouped_data = [df[dv_col][df[group_col] == g] for g in groups]

    levene_stat, levene_p = stats.levene(*grouped_data)
    print(f"   Statistic: {levene_stat:.3f}, p-value: {levene_p:.3f}")

    if levene_p > 0.05:
        print("   Conclusion: Variances appear to be equal across groups (p > 0.05).")
    else:
        print(
            "   Conclusion: Variances do not appear to be equal across groups (p <= 0.05)."
        )

    # Box plot for visual inspection
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=group_col, y=dv_col, data=df)
    plt.title("Distribution of Scores by Group")
    plt.show()
    print("\n--- End of Assumption Check ---")


def one_way_anova(df: pd.DataFrame, group_col: str, dv_col: str) -> tuple:
    """
    Performs a one-way ANOVA and Tukey's HSD post-hoc test.

    Args:
        df (pd.DataFrame): The pandas DataFrame containing the data.
        group_col (str): The name of the column containing the group labels.
        dv_col (str): The name of the column containing the values for the dependent variable.

    Returns:
        tuple: A tuple containing the ANOVA table (statsmodels ANOVA object) and the Tukey's HSD results (statsmodels TukeyHSDResults object).
    """
    # Perform one-way ANOVA
    model = ols(f"{dv_col} ~ C({group_col})", data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=1)

    # Perform Tukey's HSD post-hoc test
    tukey_results = pairwise_tukeyhsd(df[dv_col], df[group_col])

    return anova_table, tukey_results
