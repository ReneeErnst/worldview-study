import pandas as pd


def qes_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate total QES score for all columns starting with 'qes_'.
    Reverse score columns ending with '_r' before summing.
    Automatically determines max_score from the data.

    Args:
        df (pd.DataFrame): DataFrame to add the total values to.

    Returns:
        pd.DataFrame: DF with the scale total added.
    """
    # Find all qes_ columns
    qes_cols = [col for col in df.columns if col.startswith("qes")]
    
    # Determine max_score from all qes_ columns
    max_score = df[qes_cols].max().max()
    
    # Identify reverse-scored columns (ending with _r)
    reverse_cols = [col for col in qes_cols if col.endswith("r")]
    
    # Reverse score those columns
    if reverse_cols:
        df = reverse_score(df, reverse_cols, max_score)
        
    # Calculate total
    df["qes_total"] = df[qes_cols].sum(axis=1)
    
    return df


def reverse_score(df: pd.DataFrame, items: list[str], max_score: int) -> pd.DataFrame:
    """
    Reverse score specified items in a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the items.
        items (list[str]): List of column names to reverse score.
        max_score (int): Maximum value on the scale (e.g., 5 for a 1-5 scale).

    Returns:
        pd.DataFrame: DataFrame with reversed items.
    """
    for item in items:
        df[item] = max_score + 1 - df[item]
    return df


def isi_subscale_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate total scores for ISI subscales and add them as new columns.

    - Sums all columns starting with 'isi_d' into 'isi_diffuse_avoidant_total'
    - Sums all columns starting with 'isi_i' into 'isi_informational_total'
    - Sums all columns starting with 'isi_n' into 'isi_normative_total'

    Args:
        df (pd.DataFrame): DataFrame to add the subscale totals to.

    Returns:
        pd.DataFrame: DataFrame with subscale totals added.

    Raises:
        ValueError: If any subscale does not have exactly 9 items.
    """
    isi_d_cols = [col for col in df.columns if col.startswith("isi_d")]
    isi_i_cols = [col for col in df.columns if col.startswith("isi_i")]
    isi_n_cols = [col for col in df.columns if col.startswith("isi_n")]

    if len(isi_d_cols) != 9:
        raise ValueError(f"Expected 9 items for isi_diffuse_avoidant, found {len(isi_d_cols)}")
    if len(isi_i_cols) != 9:
        raise ValueError(f"Expected 9 items for isi_informational, found {len(isi_i_cols)}")
    if len(isi_n_cols) != 9:
        raise ValueError(f"Expected 9 items for isi_normative, found {len(isi_n_cols)}")

    df["isi_diffuse_avoidant_total"] = df[isi_d_cols].sum(axis=1)
    df["isi_informational_total"] = df[isi_i_cols].sum(axis=1)
    df["isi_normative_total"] = df[isi_n_cols].sum(axis=1)

    return df
