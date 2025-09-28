import warnings
import pandas as pd
import pingouin as pg
from statsmodels.formula.api import ols
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.weightstats import ttest_ind
import numpy as np
from scipy.stats import chi2_contingency
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from statsmodels.stats.multitest import multipletests


def pearsons_correlation_matrix(df_corrs: pd.DataFrame) -> pd.DataFrame:
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


def calculate_spearman_correlations(
    df: pd.DataFrame, ordinal_col: str, continuous_cols: list
) -> pd.DataFrame:
    """
    Calculates and formats Spearman correlations between one variable and a list of others.

    This function uses Pingouin to compute pairwise Spearman correlations between a single
    specified column (e.g., an ordinal variable) and a list of other columns
    (e.g., continuous variables).

    Args:
        df (pd.DataFrame): The DataFrame containing all the data.
        ordinal_col (str): The name of the single column to use as the primary variable.
        continuous_cols (list): A list of column names to correlate against the ordinal_col.

    Returns:
        pd.DataFrame: A cleaned-up DataFrame containing the specified correlation
                      coefficients, CIs, p-values, and other statistics.
    """
    # Define the list of columns we are interested in.
    all_cols = [ordinal_col] + continuous_cols

    # Calculate the pairwise correlations only for the specified columns.
    df_corr_matrix = pg.pairwise_corr(data=df[all_cols], method="spearman")

    # FIX: Filter the matrix to only include pairs with the ordinal column as 'X'.
    df_corr_matrix = df_corr_matrix[df_corr_matrix["X"] == ordinal_col].copy()

    # Calculate and add the 'df' (degrees of freedom) column
    df_corr_matrix["df"] = df_corr_matrix["n"] - 2

    # Rename common columns for clarity
    df_corr_matrix = df_corr_matrix.rename(
        columns={"n": "Sample Size", "p-unc": "p-value", "r": "rs"}
    )

    # Define the desired columns for the final output
    final_columns = ["X", "Y", "rs", "CI95%", "p-value", "df", "Sample Size"]

    # Filter for columns that actually exist in the output to avoid errors
    # (e.g., CI95% may not always be present)
    final_columns_exist = [
        col for col in final_columns if col in df_corr_matrix.columns
    ]

    return df_corr_matrix[final_columns_exist]


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


def one_way_anova(
    df: pd.DataFrame, group_col: str, dv_col: str, alpha: float = 0.05
) -> tuple:
    """
    Performs a one-way ANOVA and, if the result is significant, Tukey's HSD post-hoc test.

    Args:
        df (pd.DataFrame): The pandas DataFrame containing the data.
        group_col (str): The name of the column containing the group labels.
        dv_col (str): The name of the column containing the values for the dependent variable.
        alpha (float): The significance level to use for the ANOVA test. Defaults to 0.05.

    Returns:
        tuple: A tuple containing:
               - anova_table (pd.DataFrame): The ANOVA results table.
               - tukey_results (TukeyHSDResults or None): The Tukey's HSD results if the
                 ANOVA is significant (p < alpha), otherwise None.
    """
    # --- 1. Perform one-way ANOVA ---
    model = ols(f"{dv_col} ~ C({group_col})", data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=1)

    # --- 2. Check for significance ---
    # Get the p-value from the ANOVA table. It's in the first row for the group effect.
    p_value = anova_table["PR(>F)"].iloc[0]

    # --- 3. Conditionally perform post-hoc test ---
    tukey_results = None  # Initialize as None
    if p_value < alpha:
        tukey_results = pairwise_tukeyhsd(
            endog=df[dv_col], groups=df[group_col], alpha=alpha
        )

    return anova_table, tukey_results


def get_independent_ttest(df: pd.DataFrame, x: str, y: str) -> pd.Series:
    """
    Performs an independent two-sample t-test between two groups using statsmodels.

    It automatically chooses between a Student's t-test and a Welch's t-test.
    If the ratio of the sample sizes is more severe than 2:1, it uses a
    Welch's t-test. Otherwise, it uses a standard Student's t-test.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data.
        x (str): Independent variable (grouping variable).
        y (str): Dependent variable (continuous variable).

    Returns:
        pd.Series: contains the results, including which test was used,
              the t-statistic, p-value, and degrees of freedom.
    """
    # Create a clean DataFrame by dropping rows with NaN in either the grouping or dependent variable.
    df_clean = df[[x, y]].dropna()

    # Get the unique values of the grouping variable from the cleaned data
    x_values = df_clean[x].unique()
    if len(x_values) != 2:
        raise ValueError(
            f"'{x}' must have exactly two unique valuesafter removing NaNs. "
            f"Found {len(x_values)}: {list(x_values)}"
        )

    # Create the two arrays for comparison from the cleaned DataFrame
    group1 = df_clean[df_clean[x] == x_values[0]][y]
    group2 = df_clean[df_clean[x] == x_values[1]][y]

    n1, n2 = len(group1), len(group2)

    # This check is now mostly for safety, as dropna should prevent this.
    if n1 == 0 or n2 == 0:
        raise ValueError("One or both groups have no valid data after removing NaNs.")

    # Determine which test to use based on the sample size ratio
    ratio = max(n1, n2) / min(n1, n2)

    print(f"\nIs there a significant difference in {y} by {x}?")
    if ratio > 2:
        test_type = "unequal"
        test_name = "Welch's t-test (unequal variances)"
        print("Using Welch's t-test due to unequal sample sizes (ratio > 2:1).")
    else:
        test_type = "pooled"
        test_name = "Student's t-test (pooled variances)"
        print(
            "Using standard Student's t-test due to equal sample sizes (ratio <= 2:1)."
        )

    # Perform the independent t-test using statsmodels
    t_stat, p_value, dof = ttest_ind(group1, group2, usevar=test_type)

    if p_value < 0.05:
        print(f"There is a significant difference in {y} by {x} (p = {p_value:.3f}).")
    else:
        print(f"There is no significant difference in {y} by {x} (p = {p_value:.3f}).")

    return pd.DataFrame(
        {
            "Metric": [
                "Test Used",
                "Group Sizes",
                "T-Statistic",
                "P-Value",
                "Degrees of Freedom",
            ],
            "Value": [
                test_name,
                f"{x_values[0]}: {n1}, {x_values[1]}: {n2}",
                t_stat,
                p_value,
                dof,
            ],
        }
    )


def run_chi2_analysis(
    df: pd.DataFrame, x: str, y: str, alpha: float = 0.05, p_adjust_method: str = "holm"
) -> dict:
    """
    Performs a Chi-squared test of independence and, if significant,
    conducts pairwise post-hoc tests with p-value correction using statsmodels.

    This function returns a dictionary containing all values needed for a
    complete APA-style write-up.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        x (str): The column name for the independent variable (groups).
        y (str): The column name for the dependent variable (categories).
        alpha (float): The significance level. Defaults to 0.05.
        p_adjust_method (str): The p-value adjustment method for post-hoc tests.
                               See statsmodels.stats.multitest.multipletests
                               documentation for options. Defaults to "holm".

    Returns:
        dict: A dictionary containing the contingency table, main test results,
              expected frequencies, post-hoc results (if applicable), and an
              APA-formatted summary string.
    """
    # Create Contingency Table (Observed Frequencies) ---
    contingency_table = pd.crosstab(df[x], df[y])

    # Calculate and Format Row Percentages
    row_percentages = pd.crosstab(df[x], df[y], normalize="index") * 100

    # Create a new DataFrame to combine counts and percentages
    combined_table = contingency_table.copy()
    for col in combined_table.columns:
        combined_table[col] = (
            combined_table[col].astype(str)
            + " ("
            + row_percentages[col].round(2).astype(str)
            + "%)"
        )

    # Perform the Main Chi-Squared Test ---
    chi2, p_value, dof, expected_freqs = chi2_contingency(contingency_table)

    # Calculate Sample Size and Effect Size (Cramer's V) ---
    sample_size = contingency_table.sum().sum()
    min_dim = min(contingency_table.shape) - 1
    cramer_v = np.sqrt(chi2 / (sample_size * min_dim)) if min_dim > 0 else 0

    # Format the p-value for APA style ---
    if p_value < 0.001:
        p_apa = "< .001"
    else:
        p_apa = f"= {p_value:.3f}"

    # Create APA Summary String for the Main Test ---
    apa_summary = (
        f"χ²({dof}, N = {sample_size}) = {chi2:.2f}, "
        f"p {p_apa}, Cramer's V = {cramer_v:.2f}"
    )

    # Conditionally Perform Post-Hoc Tests ---
    posthoc_results = None
    if p_value < alpha:
        print(f"Overall test was significant (p = {p_value:.4f}).")
        print(f"Conducting post-hoc tests with {p_adjust_method} correction.")

        groups = contingency_table.index.tolist()
        categories = contingency_table.columns.tolist()

        all_pairs = list(combinations(groups, 2))

        # A list to store the results of each 2x2 test
        posthoc_data = []

        # Loop through all pairs of groups and all categories
        for group1, group2 in all_pairs:
            for category in categories:
                # Build the 2x2 contingency table for the comparison
                table_data = np.array(
                    [
                        [
                            contingency_table.loc[group1, category],
                            contingency_table.loc[group1, :].sum()
                            - contingency_table.loc[group1, category],
                        ],
                        [
                            contingency_table.loc[group2, category],
                            contingency_table.loc[group2, :].sum()
                            - contingency_table.loc[group2, category],
                        ],
                    ]
                )

                # Check for zero rows or columns to prevent ValueError
                if np.any(table_data.sum(axis=0) == 0) or np.any(
                    table_data.sum(axis=1) == 0
                ):
                    p_uncorr = 1.0  # No difference if one category has zero counts
                else:
                    # Perform the Chi-squared test on the 2x2 table
                    _, p_uncorr, _, _ = chi2_contingency(table_data, correction=False)

                # Store the uncorrected p-value and relevant info
                posthoc_data.append(
                    {
                        "group1": group1,
                        "group2": group2,
                        "category": category,
                        "p_uncorrected": p_uncorr,
                    }
                )

        # Apply p-value correction to the uncorrected p-values
        p_values_uncorrected = [d["p_uncorrected"] for d in posthoc_data]
        reject, p_values_corrected, _, _ = multipletests(
            p_values_uncorrected, alpha=alpha, method=p_adjust_method
        )

        # Update the results with corrected p-values and significance flags
        for i, (p_corr, sig) in enumerate(zip(p_values_corrected, reject)):
            posthoc_data[i]["p_corrected"] = p_corr
            posthoc_data[i]["significant"] = sig

        posthoc_results = pd.DataFrame(posthoc_data)

    else:
        print(
            f"Overall test was not significant (p = {p_value:.4f}). No post-hoc tests were performed."
        )

    # Compile All Results into a Dictionary
    results = {
        "contingency_table": combined_table,
        "main_test": {
            "chi2_statistic": chi2,
            "dof": dof,
            "p_value": p_value,
            "sample_size": sample_size,
            "cramer_v_effect_size": cramer_v,
        },
        "expected_frequencies": pd.DataFrame(
            expected_freqs,
            index=contingency_table.index,
            columns=contingency_table.columns,
        ),
        "posthoc_results": posthoc_results,
        "apa_summary": apa_summary,
    }

    return results


def run_multinomial_regression(df, iv_name, dv_name):
    """
    Performs multinomial logistic regression and prints the results,
    including odds ratios and their confidence intervals.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        iv_name (str): The name of the continuous independent variable column.
        dv_name (str): The name of the categorical dependent variable column.

    Returns:
        statsmodels.discrete.discrete_model.MNLogitResults: The fitted model results object.
    """

    # Check if the dependent variable exists in the DataFrame
    if dv_name not in df.columns:
        raise ValueError(f"Dependent variable '{dv_name}' not found in the DataFrame.")

    # Check if the independent variable exists in the DataFrame
    if iv_name not in df.columns:
        raise ValueError(
            f"Independent variable '{iv_name}' not found in the DataFrame."
        )

    # Check if the dependent variable is categorical
    if not pd.api.types.is_categorical_dtype(
        df[dv_name]
    ) and not pd.api.types.is_object_dtype(df[dv_name]):
        print(
            f"Warning: Dependent variable '{dv_name}' is not of a categorical type. Attempting to convert."
        )
        df[dv_name] = df[dv_name].astype("category")

    # Define the independent (X) and dependent (y) variables
    X = df[iv_name]
    y = df[dv_name]

    # Encode the categorical dependent variable into numerical format
    y_encoded = y.astype("category").cat.codes

    # Add a constant to the independent variable for the intercept
    X = sm.add_constant(X)

    # Fit the Multinomial Logistic Regression model
    model = sm.MNLogit(y_encoded, X)
    mnlogit_fit = model.fit(method="newton", maxiter=100)

    return mnlogit_fit
