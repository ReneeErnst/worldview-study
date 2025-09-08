import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import seaborn as sns


def check_normality(df: pd.DataFrame, dist_plot: bool = True, qq_plot: bool = True):
    """
    Checks the normality assumption of variables in a DataFrame using
    descriptive statistics, statistical tests, and visual methods.

    Args:
        df (pd.DataFrame): The DataFrame containing the variables.
        dist_plot (bool): Whether to display distribution plots (histograms).
        qq_plot (bool): Whether to display Q-Q plots.
    """

    for col in df.columns:
        print(f"*** Normality Checks for: {col} ***")

        # Descriptive Statistics
        print("\nDescriptive Statistics:")
        print(df[col].describe())
        print(f"Skewness: {df[col].skew()}")
        print(f"Kurtosis: {df[col].kurtosis()}")

        # Kolmogorov-Smirnov Test (preferred since our sample size is > 50)
        # Need to standardize the data for KS test
        ks_test = stats.kstest((df[col] - df[col].mean()) / df[col].std(), "norm")
        print(
            f"Kolmogorov-Smirnov Test: Statistic={ks_test.statistic}, p-value={ks_test.pvalue}"
        )

        # Visual Checks (if requested)
        if dist_plot or qq_plot:
            fig, ax = plt.subplots(
                1, 2 if dist_plot and qq_plot else 1, figsize=(12, 4)
            )

            if dist_plot:
                # Distribution Plot (Histogram)
                sns.histplot(df[col], kde=True, ax=ax[0] if qq_plot else ax)
                ax[0].set_title(f"Distribution Plot of {col}")

            if qq_plot:
                # Q-Q Plot
                stats.probplot(df[col], dist="norm", plot=ax[1] if dist_plot else ax)
                ax[1].set_title(f"Q-Q Plot of {col}")

            plt.tight_layout()
            plt.show()

        print("-" * 40, "\n")
