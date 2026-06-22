import pandas as pd
from huggingface_hub import hf_hub_download

from config import DATA_DIR, HF_DATASET_REPO_ID


USER_NEEDS_FILE_NAME = "user_needs.csv"
USER_NEEDS_FILE_PATH = DATA_DIR / USER_NEEDS_FILE_NAME

def load_user_needs_df_from_hf() -> pd.DataFrame:
    """
    Downloads user_needs.csv from HF.
    """
    file_path = hf_hub_download(
        repo_id=HF_DATASET_REPO_ID,
        repo_type="dataset",
        filename=USER_NEEDS_FILE_NAME,
    )
    return pd.read_csv(file_path)

def load_user_needs_df_locally() -> pd.DataFrame:
    return pd.read_csv(USER_NEEDS_FILE_PATH)


if __name__ == "__main__":
    user_needs_df = load_user_needs_df_locally()
    print(user_needs_df.head())