"""
Load data from HuggingFace.
"""

import json
import tempfile

import pandas as pd
from huggingface_hub import hf_hub_download

from config import (
    CATEGORIES_FILE_NAME,
    COMPONENTS_FILE_NAME,
    DEMAND_FILE_NAME,
    USER_NEEDS_FILE_NAME,
)

HF_DATASET_REPO_ID = "aslan-ng/bike-design-explorer"


def _download_csv(filename: str, cache: bool) -> pd.DataFrame:
    if cache:
        file_path = hf_hub_download(
            repo_id=HF_DATASET_REPO_ID,
            repo_type="dataset",
            filename=filename,
        )
        return pd.read_csv(file_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = hf_hub_download(
            repo_id=HF_DATASET_REPO_ID,
            repo_type="dataset",
            filename=filename,
            cache_dir=tmpdir,
        )
        return pd.read_csv(file_path)

def _download_json(filename: str, cache: bool) -> dict:
    if cache:
        file_path = hf_hub_download(
            repo_id=HF_DATASET_REPO_ID,
            repo_type="dataset",
            filename=filename,
        )
        with open(file_path) as f:
            return json.load(f)

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = hf_hub_download(
            repo_id=HF_DATASET_REPO_ID,
            repo_type="dataset",
            filename=filename,
            cache_dir=tmpdir,
        )
        with open(file_path) as f:
            return json.load(f)

def load_categories_df_from_hf(cache: bool = True) -> pd.DataFrame:
    return _download_csv(CATEGORIES_FILE_NAME, cache)

def load_components_df_from_hf(cache: bool = True) -> pd.DataFrame:
    return _download_csv(COMPONENTS_FILE_NAME, cache)

def load_demand_parameters_from_hf(cache: bool = True) -> dict:
    return _download_json(DEMAND_FILE_NAME, cache)

def load_user_needs_df_from_hf(cache: bool = True) -> pd.DataFrame:
    return _download_csv(USER_NEEDS_FILE_NAME, cache)


if __name__ == "__main__":
    # Check loading datasets from hf and being equal to the local
    import os
    from pandas.testing import assert_frame_equal
    from dotenv import load_dotenv

    from load_local import (
        load_categories_df_locally,
        load_components_df_locally,
        load_demand_parameters_locally,
        load_user_needs_df_locally,
    )

    load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN")

    print("Comparing local files with Hugging Face files...")

    assert_frame_equal(
        load_categories_df_locally(),
        load_categories_df_from_hf(cache=False),
        check_dtype=False,
    )
    assert_frame_equal(
        load_components_df_locally(),
        load_components_df_from_hf(cache=False),
        check_dtype=False,
    )
    assert_frame_equal(
        load_user_needs_df_locally(),
        load_user_needs_df_from_hf(cache=False),
        check_dtype=False,
    )
    assert (
        load_demand_parameters_locally()
        == load_demand_parameters_from_hf(cache=False)
    )

    print("✓ Categories match")
    print("✓ Components match")
    print("✓ User needs match")
    print("✓ Demand parameters match")

    print("✓ All local and Hugging Face data are identical.")