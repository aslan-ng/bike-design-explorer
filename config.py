from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


CUSTOMER_RESEARCH_FILE = DATA_DIR / "customer_research.csv"
DEMAND_PARAMETERS_FILE = DATA_DIR / "demand_parameters.json"

HF_DATASET_REPO_ID = "aslan-ng/bike-design-explorer"