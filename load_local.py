"""
Load data from local files.
"""

from pathlib import Path
import json
import pandas as pd

from config import (
    CATEGORIES_FILE_NAME,
    COMPONENTS_FILE_NAME,
    DEMAND_FILE_NAME,
    USER_NEEDS_FILE_NAME,
)


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

CATEGORIES_FILE_PATH = DATA_DIR / CATEGORIES_FILE_NAME
COMPONENTS_FILE_PATH = DATA_DIR / COMPONENTS_FILE_NAME
DEMAND_FILE_PATH = DATA_DIR / DEMAND_FILE_NAME
USER_NEEDS_FILE_PATH = DATA_DIR / USER_NEEDS_FILE_NAME


def load_categories_df_locally() -> pd.DataFrame:
    return pd.read_csv(CATEGORIES_FILE_PATH)

def load_components_df_locally() -> pd.DataFrame:
    return pd.read_csv(COMPONENTS_FILE_PATH)

def load_demand_parameters_locally() -> dict:
    with open(DEMAND_FILE_PATH, "r") as f:
        return json.load(f)
    
def load_user_needs_df_locally() -> pd.DataFrame:
    return pd.read_csv(USER_NEEDS_FILE_PATH)


if __name__ == "__main__":
    # Check loading datasets locally
    print("\n=== Categories ===")
    print(load_categories_df_locally().head())

    print("\n=== Components ===")
    print(load_components_df_locally().head())

    print("\n=== Demand Parameters ===")
    print(json.dumps(load_demand_parameters_locally(), indent=4))

    print("\n=== User Needs ===")
    print(load_user_needs_df_locally().head())
