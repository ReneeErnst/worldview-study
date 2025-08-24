from pathlib import Path

import numpy as np
import pandas as pd


def get_data_dir() -> Path:
    """Get path to data directory in repo."""
    return Path(__file__).parents[1] / "data"


def set_pd_display():
    """Show full dataframes in display."""
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", lambda x: "%.5f" % x)
    np.set_printoptions(threshold=100, suppress=True)
