"""
Data loading utilities for BikeDesignLab.

Place this file next to main.py, components.csv, and customer_research.csv.

Expected files:
- components.csv
- customer_research.csv
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
from huggingface_hub import (
    hf_hub_download,
    HfApi
)
from HF import *

api = HfApi(token=HF_TOKEN)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

CATEGORIES_FILE = DATA_DIR / "categories.csv"
COMPONENTS_FILE = DATA_DIR / "components.csv"
CUSTOMER_RESEARCH_FILE = DATA_DIR / "customer_research.csv"
DEMAND_PARAMETERS_FILE = DATA_DIR / "demand_parameters.json"


CUSTOMER_METADATA = {
    "Need",
}

COMPONENT_METADATA = {
    "component_category", # Category
    "component_name", # Name
    "component_description", # Description
    "component_cost", # Cost
}

def upload_customer_research_df(
    file_path: str | Path,
) -> None:
    """
    Uploads customer_research.csv to the HF dataset repo.
    """
    api.upload_file(
        path_or_fileobj=str(file_path),
        path_in_repo="customer_research.csv",
        repo_id=HF_DATASET_REPO_ID,
        repo_type="dataset",
    )

def upload_components_df(
    file_path: str | Path,
) -> None:
    """
    Uploads components.csv to the HF dataset repo.
    """
    api.upload_file(
        path_or_fileobj=str(file_path),
        path_in_repo="components.csv",
        repo_id=HF_DATASET_REPO_ID,
        repo_type="dataset",
    )

def load_customer_research_df_from_hf() -> pd.DataFrame:
    """
    Downloads customer_research.csv from HF.
    """
    file_path = hf_hub_download(
        repo_id=HF_DATASET_REPO_ID,
        repo_type="dataset",
        filename="customer_research.csv",
    )
    return pd.read_csv(file_path)

def load_components_df_from_hf() -> pd.DataFrame:
    """
    Downloads components.csv from HF.
    """
    file_path = hf_hub_download(
        repo_id=HF_DATASET_REPO_ID,
        repo_type="dataset",
        filename="components.csv",
    )
    return pd.read_csv(file_path)

def load_customer_research_df_locally(file_path: str | Path) -> pd.DataFrame:
    return pd.read_csv(file_path)

def load_components_df_locally(file_path: str | Path) -> pd.DataFrame:
    return pd.read_csv(file_path)

def get_customer_features(
    customer_research_df: pd.DataFrame,
) -> list[str]:
    """
    Returns all feature names from customer_research.csv.
    """
    df = customer_research_df
    return [
        feature
        for feature in df["feature"].tolist()
        if feature not in CUSTOMER_METADATA
    ]

def get_component_features(
    components_df: pd.DataFrame,
) -> list[str]:
    """
    Returns all feature columns from components.csv.
    """
    df = components_df
    return [
        column
        for column in df.columns
        if column not in COMPONENT_METADATA
    ]

def check_data(
    customer_research_df: pd.DataFrame,
    components_df: pd.DataFrame,
):
    """
    Check datasets for compatibility
    """
    # Load
    customer_features = get_customer_features(customer_research_df)
    components_features = get_component_features(components_df)

    # Check for duplicate entries in each
    if customer_research_df.duplicated().any():
        raise ValueError("Duplicate entries found in customer_research.csv")
    if components_df.duplicated().any():
        raise ValueError("Duplicate entries found in components.csv")

    # Checks that the features in customer_research.csv match the features in components.csv
    customer_features = set(customer_features)
    components_features = set(components_features)
    if customer_features != components_features:
        missing_from_components = customer_features - components_features
        missing_from_customer = components_features - customer_features
        raise ValueError(
            "Customer research and components data are not compatible.\n"
            f"Missing from components.csv: {missing_from_components}\n"
            f"Missing from customer_research.csv: {missing_from_customer}"
        )

def get_features(customer_research_df: pd.DataFrame) -> set[str]:
    return set(get_customer_features(customer_research_df))

def get_categories(components_df: pd.DataFrame) -> list[str]:
    return sorted(components_df["component_category"].unique().tolist())

def get_groupings(components_df: pd.DataFrame) -> list[str]:
    return sorted(components_df["component_grouping"].unique().tolist())

def get_components_name(components_df: pd.DataFrame) -> list[str]:
    return sorted(components_df["component_name"].tolist())

def get_customer_importance(customer_research_df: pd.DataFrame,) -> dict[str, float]:
    """
    Converts customer_research.csv into:
    {
        "cargo_capacity": 130,
        "climbing_ability": 150,
        ...
    }
    Assumes customer_research.csv has columns:
    - feature
    - importance
    """
    if "importance" not in customer_research_df.columns:
        raise ValueError(
            "customer_research.csv must contain an 'importance' column."
        )

    return dict(
        zip(
            customer_research_df["feature"],
            customer_research_df["importance"],
        )
    )

def get_components_in_category(
    component_category: str,
    components_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Returns component_name and component_description
    for all components in the given category.
    """
    filtered_df = components_df[
        components_df["component_category"] == component_category
    ]
    return filtered_df[
        [
            "component_name",
            "component_description",
            "component_cost"
        ]
    ].copy()

def upload():
    customer_research_df = load_customer_research_df_locally(DEFAULT_CUSTOMER_RESEARCH_FILE)
    components_df = load_components_df_locally(DEFAULT_COMPONENTS_FILE)
    check_data(
        customer_research_df=customer_research_df,
        components_df=components_df,
    )
    upload_customer_research_df(file_path=DEFAULT_CUSTOMER_RESEARCH_FILE)
    upload_components_df(file_path=DEFAULT_COMPONENTS_FILE)
    

if __name__ == "__main__":
    # Upload data
    #upload()

    # Load data locally
    customer_research_df = load_customer_research_df_locally(DEFAULT_CUSTOMER_RESEARCH_FILE)
    components_df = load_components_df_locally(DEFAULT_COMPONENTS_FILE)

    importance = get_customer_importance(customer_research_df)
    #print(importance)

    features = get_features(customer_research_df)
    #print(features)

    categories = get_categories(components_df)
    #print(categories)

    components_in_category = get_components_in_category("Frame", components_df)
    #print(components_in_category)

    groupings = get_groupings(components_df)
    #print(groupings)

    components_name = get_components_name(components_df)
    #print(components_name)

    check_data(customer_research_df, components_df)