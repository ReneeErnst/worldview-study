import pandas as pd

import worldview


def demographics_mapping(df: pd.DataFrame) -> pd.DataFrame:
    """
    Maps demographic columns to more descriptive values.

    Args:
        df (pd.DataFrame): DataFrame containing demographic columns.

    Returns:
        pd.DataFrame: DataFrame with mapped demographic values.
    """
    df_out = df.copy()

    age = {1: "25-34", 2: "35-44", 3: "45-54", 4: "55-64", 6: "65-74", 7: "75+"}

    gender = {1: "female", 2: "male", 3: "nonbinary"}

    ethnicity = {
        1: "American Indian or Alaska Native",
        2: "Asian or Asian American",
        3: "Black or African American",
        4: "Hispanic or Latino",
        5: "Middle Eastern or North African",
        6: "Native Hawaiian or other Pacific Islander",
        7: "White",
    }

    education = {
        1: "Highschool gradate or proficiency",
        2: "Attended trade school/certifications",
        3: "1-2 years college/associate’s degree",
        4: "Graduated with Bachelors",
        5: "Some graduate school",
        6: "Graduated with master’s degree",
        7: "Graduated with PhD",
    }

    religion = {
        1: "Spiritually eclectic",
        2: "Agnostic",
        3: "Atheist",
        4: "Christian",
        5: "Judaism",
        6: "Buddhist",
        7: "Hindu",
        9: "Other (please specify)",
    }

    df_out["age_group_ordinal"] = df_out["age_group"]
    df_out["age_group"] = df_out["age_group"].map(age)
    df_out["gender"] = df_out["gender"].map(gender)
    df_out["ethnicity"] = df_out["ethnicity"].map(ethnicity)
    df_out["education"] = df_out["education"].map(education)
    df_out["religious_spiritual_orientation"] = df_out[
        "religious_spiritual_orientation"
    ].map(religion)
    df_out["feel_experience_changed"] = df_out["feel_experience_changed"].map(
        {1: "yes", 2: "no"}
    )
    df_out["consider_open_inclusive"] = df_out["consider_open_inclusive"].map(
        {1: "yes", 2: "no"}
    )
    df_out["experience_open_inclusive"] = df_out["experience_open_inclusive"].map(
        {1: "yes", 2: "no"}
    )

    # rename column "transexual" to "transsexual" and clean values
    df_out = df_out.rename(columns={"transexual": "transsexual"})

    # Clean up values - "no", "No ", " NO" -> "no"
    df_out["transsexual"] = df_out["transsexual"].str.strip().str.lower()

    return df_out


def clean_demographics(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans and standardizes demographic data.
    Args:
        df (pd.DataFrame): DataFrame containing demographic columns.
    Returns:
        pd.DataFrame: DataFrame with cleaned demographic values.
    """
    df_out = df.copy()

    # If they entered "Muslim" in the other field, change their main religion to Muslim
    df_out.loc[
        df_out["religious_spiritual_orientation_other"].str.strip().str.lower()
        == "muslim",
        "religious_spiritual_orientation",
    ] = "Muslim"

    # Check the other education field for keywords and update the main education field as needed
    df_out.loc[
        df_out["education_other"].str.contains("Masters", na=False),
        "education",
    ] = "Graduated with master’s degree"

    df_out.loc[
        df_out["education_other"].str.contains("JD", na=False),
        "education",
    ] = "Graduated with PhD"
    df_out.loc[
        df_out["education_other"].str.contains("MD", na=False),
        "education",
    ] = "Graduated with PhD"

    return df_out


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
        raise ValueError(
            f"Expected 9 items for isi_diffuse_avoidant, found {len(isi_d_cols)}"
        )
    if len(isi_i_cols) != 9:
        raise ValueError(
            f"Expected 9 items for isi_informational, found {len(isi_i_cols)}"
        )
    if len(isi_n_cols) != 9:
        raise ValueError(f"Expected 9 items for isi_normative, found {len(isi_n_cols)}")

    df["isi_diffuse_avoidant_total"] = df[isi_d_cols].sum(axis=1)
    df["isi_informational_total"] = df[isi_i_cols].sum(axis=1)
    df["isi_normative_total"] = df[isi_n_cols].sum(axis=1)

    return df


def worldview_transformations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms raw worldview scores based on a predefined mapping.

    Finds all columns starting with 'wvs_', maps their values using
    {4: 1, 3: 0, 2: 0, 1: -1}, and adds them back to the DataFrame in
    new columns prefixed with 'transformed_'.

    Args:
        df: The input DataFrame with raw 'wvs_' columns.

    Returns:
        The DataFrame with added 'transformed_' columns.
    """
    df_out = df.copy()
    wvs_cols = [col for col in df_out.columns if col.startswith("wvs_")]
    score_map = {4: 1, 3: 0, 2: 0, 1: -1}

    for col in wvs_cols:
        transformed_col_name = f"transformed_{col}"
        df_out[transformed_col_name] = df_out[col].map(score_map).fillna(0)

    return df_out


def worldview_subscales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the four worldview subscale total scores.

    Sums the 'transformed_' columns based on their suffix (t, m, p, i)
    to create the final total scores. Assumes transform_scores() has
    already been run.

    Args:
        df: DataFrame containing the 'transformed_' columns.

    Returns:
        The DataFrame with added total score columns.
    """
    df_out = df.copy()
    transformed_cols = [c for c in df_out.columns if c.startswith("transformed_wvs_")]

    df_out["traditional_wvs_total"] = df_out[
        [c for c in transformed_cols if c.endswith("t")]
    ].sum(axis=1)
    df_out["modern_wvs_total"] = df_out[
        [c for c in transformed_cols if c.endswith("m")]
    ].sum(axis=1)
    df_out["postmodern_wvs_total"] = df_out[
        [c for c in transformed_cols if c.endswith("p")]
    ].sum(axis=1)
    df_out["integrative_wvs_total"] = df_out[
        [c for c in transformed_cols if c.endswith("i")]
    ].sum(axis=1)

    return df_out


def assign_dominant_worldview(df: pd.DataFrame) -> pd.DataFrame:
    """
    Determines the dominant worldview and handles ties.

    Finds the worldview with the highest score for each row. If there is a tie
    for the highest score, it assigns 'multiple_ties'. Assumes
    calculate_subscale_totals() has already been run.

    Args:
        df: DataFrame containing the four '_wvs_total' columns.

    Returns:
        The DataFrame with the final 'dominant_worldview' column.
    """
    df_out = df.copy()
    worldview_cols = [
        "traditional_wvs_total",
        "modern_wvs_total",
        "postmodern_wvs_total",
        "integrative_wvs_total",
    ]

    # Identify ties for the max value in each row
    max_values = df_out[worldview_cols].max(axis=1)
    is_max = df_out[worldview_cols].eq(max_values, axis=0)
    max_counts = is_max.sum(axis=1)

    tie_count = (max_counts > 1).sum()
    print(f"Number of rows with a tie for the maximum worldview score: {tie_count}")

    # Assign the dominant worldview
    df_out["dominant_worldview"] = (
        df_out[worldview_cols].idxmax(axis=1).str.replace("_wvs_total", "", regex=False)
    )
    df_out.loc[max_counts > 1, "dominant_worldview"] = "multiple_ties"

    return df_out


def calculate_worldview_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs the full pipeline to calculate worldview scores and dominant worldview.

    This function chains together the transformation, subscale calculation,
    and dominant worldview assignment steps.

    Args:
        df: A pandas DataFrame containing the raw survey data.

    Returns:
        A pandas DataFrame with all calculated columns appended.
    """
    processed_df = (
        df.pipe(worldview_transformations)
        .pipe(worldview_subscales)
        .pipe(assign_dominant_worldview)
    )
    return processed_df


def create_prepped_data() -> pd.DataFrame:
    """
    Creates prepped/preprocessed data with all transformations and calculations applied.

    Returns:
        pd.DataFrame: The fully prepped/processed DataFrame.
    """
    data_loc = worldview.get_data_dir()
    df = pd.read_excel(data_loc / "prepped_raw_data.xlsx", header=0, skiprows=[1])

    df_prepped = (
        df.pipe(demographics_mapping)
        .pipe(clean_demographics)
        .pipe(qes_scores)
        .pipe(isi_subscale_scores)
        .pipe(calculate_worldview_scores)
    )

    df_prepped.to_csv(data_loc / "prepped_full_data.csv")

    keep_cols = [
        "age_group",
        "age_group_ordinal",
        "gender",
        "transsexual",
        "ethnicity",
        "education",
        "religious_spiritual_orientation",
        "feel_experience_changed",
        "consider_open_inclusive",
        "experience_open_inclusive",
        "traditional_wvs_total",
        "modern_wvs_total",
        "postmodern_wvs_total",
        "integrative_wvs_total",
        "dominant_worldview",
        "qes_total",
        "isi_diffuse_avoidant_total",
        "isi_informational_total",
        "isi_normative_total",
    ]

    df_prepped[keep_cols].to_csv(data_loc / "prepped_filtered_data.csv")

    return df_prepped
