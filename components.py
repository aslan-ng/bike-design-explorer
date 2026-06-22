import pandas as pd
from huggingface_hub import hf_hub_download

from config import DATA_DIR, HF_DATASET_REPO_ID


COMPONENTS_FILE_NAME = "components.csv"
COMPONENTS_FILE_PATH = DATA_DIR / COMPONENTS_FILE_NAME


def load_components_df_from_hf() -> pd.DataFrame:
    """
    Downloads components.csv from HF.
    """
    file_path = hf_hub_download(
        repo_id=HF_DATASET_REPO_ID,
        repo_type="dataset",
        filename=COMPONENTS_FILE_NAME,
    )
    return pd.read_csv(file_path)

def load_components_df_locally() -> pd.DataFrame:
    return pd.read_csv(COMPONENTS_FILE_PATH)


if __name__ == "__main__":
    components_df = load_components_df_locally()
    print(components_df.head())