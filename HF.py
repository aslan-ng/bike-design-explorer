HF_DATASET_REPO_ID = "aslan-ng/bike-design-explorer"
HF_TOKEN = "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"


if __name__ == "__main__":

    from huggingface_hub import HfApi

    api = HfApi(token=HF_TOKEN)
    print(api.whoami())