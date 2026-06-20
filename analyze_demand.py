from itertools import product
import pandas as pd
from utils import *
from data_loader import *
import json
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
from HF import *

BASE_DIR = Path(__file__).resolve().parent
DEMAND_PARAMETERS_FILE = "demand_parameters.json"
DEMAND_PARAMETERS_FILE_LOCAL = BASE_DIR / "demand_parameters.json"
HF_DATASET_REPO_ID = "aslan-ng/bike-design-explorer"


def save_demand_parameters_locally(
    demand_parameters: dict,
    file_path: str | Path = DEMAND_PARAMETERS_FILE_LOCAL,
) -> None:
    with open(file_path, "w") as f:
        json.dump(demand_parameters, f, indent=4)

def load_demand_parameters_locally(
    file_path: str | Path = DEMAND_PARAMETERS_FILE,
) -> dict:
    with open(file_path, "r") as f:
        return json.load(f)

def upload_demand_parameters_to_hf(
    file_path: str | Path = DEMAND_PARAMETERS_FILE,
) -> None:
    api = HfApi(token=HF_TOKEN)
    api.upload_file(
        path_or_fileobj=str(file_path),
        path_in_repo=DEMAND_PARAMETERS_FILE,
        repo_id=HF_DATASET_REPO_ID,
        repo_type="dataset",
    )

def load_demand_parameters_from_hf(
    token: str | None = None,
) -> dict:
    file_path = hf_hub_download(
        repo_id=HF_DATASET_REPO_ID,
        repo_type="dataset",
        filename=DEMAND_PARAMETERS_FILE,
        token=token,
    )
    with open(file_path, "r") as f:
        return json.load(f)

def estimate_demand_parameters(
    customer_research_df: pd.DataFrame,
    components_df: pd.DataFrame,
    markup: float = 1.5,
    base_market_size: int = 100,
) -> dict:
    """
    Estimates demand-function parameters from the available design space.

    Assumes:
    - one component must be selected from each component_grouping
    - selling price is roughly markup * design cost
    """

    groupings = get_groupings(components_df)

    component_options_by_grouping = [
        components_df[
            components_df["component_grouping"] == grouping
        ]
        for grouping in groupings
    ]

    all_designs = []

    for design_parts in product(*[
        group_df.to_dict(orient="records")
        for group_df in component_options_by_grouping
    ]):
        selected_components = pd.DataFrame(design_parts)

        cost = calculate_design_cost(selected_components)

        satisfaction = calculate_design_satisfaction(
            selected_components=selected_components,
            customer_research_df=customer_research_df,
        )

        all_designs.append(
            {
                "cost": cost,
                "selling_price": cost * markup,
                "satisfaction": satisfaction,
            }
        )

    design_space_df = pd.DataFrame(all_designs)

    reference_satisfaction = float(
        design_space_df["satisfaction"].median()
    )

    reference_price = float(
        design_space_df["selling_price"].median()
    )

    # Makes demand about 50% of base market size at the reference price.
    price_sensitivity = 0.5 / reference_price

    return {
        "base_market_size": base_market_size,
        "reference_satisfaction": reference_satisfaction,
        "reference_price": reference_price,
        "price_sensitivity": price_sensitivity,
        "design_space_summary": {
            "number_of_possible_designs": len(design_space_df),
            "min_cost": float(design_space_df["cost"].min()),
            "median_cost": float(design_space_df["cost"].median()),
            "max_cost": float(design_space_df["cost"].max()),
            "min_satisfaction": float(design_space_df["satisfaction"].min()),
            "median_satisfaction": float(design_space_df["satisfaction"].median()),
            "max_satisfaction": float(design_space_df["satisfaction"].max()),
        },
    }


if __name__ == "__main__":
    customer_research_df = load_customer_research_df_locally(DEFAULT_CUSTOMER_RESEARCH_FILE)
    components_df = load_components_df_locally(DEFAULT_COMPONENTS_FILE)
    demand_parameters = estimate_demand_parameters(
        customer_research_df=customer_research_df,
        components_df=components_df,
    )
    #print(demand_parameters)

    save_demand_parameters_locally(demand_parameters)

    upload_demand_parameters_to_hf(file_path=DEMAND_PARAMETERS_FILE_LOCAL)