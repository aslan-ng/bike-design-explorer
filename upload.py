from config import HF_DATASET_REPO_ID
from categories import (
    CATEGORIES_FILE_NAME,
    CATEGORIES_FILE_PATH
)
from components import (
    COMPONENTS_FILE_NAME,
    COMPONENTS_FILE_PATH
)
from user_needs import (
    USER_NEEDS_FILE_NAME,
    USER_NEEDS_FILE_PATH
)


def upload(api):
    # Categories
    api.upload_file(
        path_or_fileobj=str(CATEGORIES_FILE_PATH),
        path_in_repo=CATEGORIES_FILE_NAME,
        repo_id=HF_DATASET_REPO_ID,
        repo_type="dataset",
    )
    # Components
    api.upload_file(
        path_or_fileobj=str(COMPONENTS_FILE_PATH),
        path_in_repo=COMPONENTS_FILE_NAME,
        repo_id=HF_DATASET_REPO_ID,
        repo_type="dataset",
    )
    # User Needs
    api.upload_file(
        path_or_fileobj=str(USER_NEEDS_FILE_PATH),
        path_in_repo=USER_NEEDS_FILE_NAME,
        repo_id=HF_DATASET_REPO_ID,
        repo_type="dataset",
    )


if __name__ == "__main__":

    import os
    from dotenv import load_dotenv
    from huggingface_hub import HfApi
    
    load_dotenv()

    HF_TOKEN = os.getenv("HF_TOKEN")
    api = HfApi(token=HF_TOKEN)
    print(api.whoami())

    upload(api)