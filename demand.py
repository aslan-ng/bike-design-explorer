#from analyze_demand import *
import json
from huggingface_hub import hf_hub_download

from config import DATA_DIR, HF_DATASET_REPO_ID


DEMAND_FILE_NAME = "demand.json"
DEMAND_FILE_PATH = DATA_DIR / DEMAND_FILE_NAME


def load_demand_parameters_locally() -> dict:
    with open(DEMAND_FILE_PATH, "r") as f:
        return json.load(f)

def load_demand_parameters_from_hf() -> dict:
    file_path = hf_hub_download(
        repo_id=HF_DATASET_REPO_ID,
        repo_type="dataset",
        filename=DEMAND_FILE_NAME,
    )
    with open(file_path, "r") as f:
        return json.load(f)




'''
def calculate_demand(
    selling_price: float,
    satisfaction: float,
    demand_parameters: dict | None = None,
) -> int:
    """
    Calculates demand using calibrated demand parameters.
    Demand increases with satisfaction.
    Demand decreases with selling price.
    """
    if demand_parameters is None:
        demand_parameters = load_demand_parameters_locally() # Locally
    base_market_size = demand_parameters["base_market_size"]
    reference_satisfaction = demand_parameters["reference_satisfaction"]
    price_sensitivity = demand_parameters["price_sensitivity"]
    satisfaction_effect = satisfaction / reference_satisfaction
    price_effect = max(
        0,
        1 - price_sensitivity * selling_price,
    )

    demand = (
        base_market_size
        * satisfaction_effect
        * price_effect
    )
    return max(0, int(round(demand)))
'''
def calculate_profit(
    selling_price: float,
    design_cost: float,
    demand: int,
) -> float:
    return float((selling_price - design_cost) * demand)


if __name__ == "__main__":
    demand_df = load_demand_parameters_locally()
    print(demand_df)

