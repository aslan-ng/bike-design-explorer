import pandas as pd
from huggingface_hub import hf_hub_download

from config import DATA_DIR, HF_DATASET_REPO_ID


CATEGORIES_FILE_NAME = "categories.csv"
CATEGORIES_FILE_PATH = DATA_DIR / CATEGORIES_FILE_NAME

def load_categories_df_from_hf() -> pd.DataFrame:
    """
    Downloads categories.csv from HF.
    """
    file_path = hf_hub_download(
        repo_id=HF_DATASET_REPO_ID,
        repo_type="dataset",
        filename=CATEGORIES_FILE_NAME,
    )
    return pd.read_csv(file_path)

def load_categories_df_locally() -> pd.DataFrame:
    return pd.read_csv(CATEGORIES_FILE_PATH)


if __name__ == "__main__":
    categories_df = load_categories_df_locally()
    print(categories_df.head())