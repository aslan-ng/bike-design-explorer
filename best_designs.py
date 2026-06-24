import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from load_local import DATA_DIR, DEMAND_FILE_PATH
from calculate_demand_parameters import (
    COST_ARRAY_PATH,
    SATISFACTION_ARRAY_PATH,
    OPTIONS_PATH,
)

BEST_DESIGNS_DIR = DATA_DIR / "best_designs"
output_path = BEST_DESIGNS_DIR / "best_designs_by_objective.csv"

def load_demand_parameters() -> dict:
    with open(DEMAND_FILE_PATH, "r") as f:
        return json.load(f)


def load_category_options() -> list[dict]:
    with open(OPTIONS_PATH, "rb") as f:
        return pickle.load(f)


def index_to_option_indices(index: int, option_counts: list[int]) -> list[int]:
    option_indices = [0] * len(option_counts)

    for i in range(len(option_counts) - 1, -1, -1):
        option_indices[i] = index % option_counts[i]
        index //= option_counts[i]

    return option_indices


def build_selected_components_from_index(
    design_index: int,
    category_options: list[dict],
) -> dict[str, list[str]]:
    option_counts = [
        len(category_data["options"])
        for category_data in category_options
    ]

    option_indices = index_to_option_indices(
        index=design_index,
        option_counts=option_counts,
    )

    selected_components = {}

    for category_index, option_index in enumerate(option_indices):
        option = category_options[category_index]["options"][option_index]

        if option["component_names"]:
            selected_components[option["category"]] = option["component_names"]

    return selected_components


def calculate_demand_array(
    selling_prices: np.ndarray,
    satisfactions: np.ndarray,
    demand_parameters: dict,
) -> np.ndarray:
    base_market_size = demand_parameters["base_market_size"]
    reference_satisfaction = demand_parameters["reference_satisfaction"]
    reference_price = demand_parameters["reference_price"]
    max_satisfaction = demand_parameters["max_satisfaction"]
    satisfaction_weight = demand_parameters["satisfaction_weight"]
    price_sensitivity = demand_parameters["price_sensitivity"]

    normalized_satisfaction = satisfactions / max_satisfaction

    satisfaction_effect = (
        normalized_satisfaction / reference_satisfaction
    ) ** satisfaction_weight

    price_effect = np.maximum(
        0.0,
        1.0 - price_sensitivity * (selling_prices - reference_price),
    )

    demand = base_market_size * satisfaction_effect * price_effect

    return np.maximum(0, np.rint(demand)).astype(int)


def get_top_indices(values: np.ndarray, top_k: int = 10) -> np.ndarray:
    top_k = min(top_k, len(values))
    candidate_indices = np.argpartition(values, -top_k)[-top_k:]
    sorted_indices = candidate_indices[np.argsort(values[candidate_indices])[::-1]]
    return sorted_indices


def summarize_designs(
    indices: np.ndarray,
    objective_values: np.ndarray,
    costs: np.ndarray,
    satisfactions: np.ndarray,
    selling_prices: np.ndarray,
    demands: np.ndarray,
    profits: np.ndarray,
    category_options: list[dict],
    objective_name: str,
) -> pd.DataFrame:
    rows = []

    for rank, design_index in enumerate(indices, start=1):
        rows.append(
            {
                "rank": rank,
                "objective": objective_name,
                "design_index": int(design_index),
                "objective_value": float(objective_values[design_index]),
                "cost": float(costs[design_index]),
                "selling_price": float(selling_prices[design_index]),
                "user_satisfaction": float(satisfactions[design_index]),
                "demand": int(demands[design_index]),
                "profit": float(profits[design_index]),
                "selected_components": build_selected_components_from_index(
                    design_index=int(design_index),
                    category_options=category_options,
                ),
            }
        )

    return pd.DataFrame(rows)


def analyze_best_designs(
    top_k: int = 10,
    markup: float = 1.5,
) -> pd.DataFrame:
    demand_parameters = load_demand_parameters()
    category_options = load_category_options()

    costs = np.load(COST_ARRAY_PATH, mmap_mode="r")
    satisfactions = np.load(SATISFACTION_ARRAY_PATH, mmap_mode="r")

    selling_prices = costs * markup

    demands = calculate_demand_array(
        selling_prices=selling_prices,
        satisfactions=satisfactions,
        demand_parameters=demand_parameters,
    )

    profits = (selling_prices - costs) * demands

    objectives = {
        "maximize_user_satisfaction": satisfactions,
        "maximize_demand": demands,
        "maximize_profit": profits,
        "maximize_satisfaction_per_cost": satisfactions / costs,
        "maximize_satisfaction_per_price": satisfactions / selling_prices,
    }

    all_results = []

    for objective_name, objective_values in objectives.items():
        top_indices = get_top_indices(objective_values, top_k=top_k)

        result_df = summarize_designs(
            indices=top_indices,
            objective_values=objective_values,
            costs=costs,
            satisfactions=satisfactions,
            selling_prices=selling_prices,
            demands=demands,
            profits=profits,
            category_options=category_options,
            objective_name=objective_name,
        )

        all_results.append(result_df)

    return pd.concat(all_results, ignore_index=True)


if __name__ == "__main__":
    results_df = analyze_best_designs(
        top_k=10,
        markup=1.5,
    )

    pd.set_option("display.max_colwidth", None)
    print(results_df)

    BEST_DESIGNS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    print(f"\nSaved to: {output_path}")