"""
Analyze all configurations and create demand_parameters.json.

Fast version:
- Precomputes cost/satisfaction for each category option.
- Avoids pandas inside the 55,987,200-design loop.
- Uses multiprocessing by index chunks.
- Saves resumable checkpoints.
"""

import os
import json
import math
import pickle
from pathlib import Path
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from tqdm import tqdm

from load_local import DATA_DIR, DEMAND_FILE_PATH
from design import calculate_user_satisfaction, calculate_design_cost


CHECKPOINT_DIR = DATA_DIR / "demand_analysis"
CHECKPOINT_STATE_PATH = CHECKPOINT_DIR / "checkpoint_state.json"
COST_ARRAY_PATH = CHECKPOINT_DIR / "costs.npy"
SATISFACTION_ARRAY_PATH = CHECKPOINT_DIR / "satisfactions.npy"
OPTIONS_PATH = CHECKPOINT_DIR / "precomputed_options.pkl"


_WORKER_OPTIONS = None
_WORKER_OPTION_COUNTS = None
_WORKER_TOTAL_CATEGORIES = None
_WORKER_MARKUP = None


def save_demand_parameters_locally(
    demand_parameters: dict,
    file_path: str | Path = DEMAND_FILE_PATH,
) -> None:
    with open(file_path, "w") as f:
        json.dump(demand_parameters, f, indent=4)

def get_category_selection_options(
    category: str,
    components_df: pd.DataFrame,
    required: bool,
    multiple_allowed: bool,
) -> list[list[str]]:
    component_names = (
        components_df.loc[
            components_df["component_category"] == category,
            "component_name",
        ]
        .dropna()
        .tolist()
    )

    if required:
        return [[name] for name in component_names]

    if multiple_allowed:
        options = [[]]

        for r in range(1, len(component_names) + 1):
            for combo in combinations(component_names, r):
                options.append(list(combo))

        return options

    return [[]] + [[name] for name in component_names]

def precompute_category_options(
    components_df: pd.DataFrame,
    categories_df: pd.DataFrame,
    user_needs_df: pd.DataFrame,
) -> list[dict]:
    category_options = []

    for _, category_row in categories_df.iterrows():
        category = category_row["categories"]
        required = bool(category_row["required"])
        multiple_allowed = bool(category_row["multiple_allowed"])

        raw_options = get_category_selection_options(
            category=category,
            components_df=components_df,
            required=required,
            multiple_allowed=multiple_allowed,
        )

        options = []

        for component_names in raw_options:
            selected_components_df = components_df.loc[
                (components_df["component_category"] == category)
                & (components_df["component_name"].isin(component_names))
            ]

            cost = calculate_design_cost(selected_components_df)

            if len(selected_components_df) == 0:
                satisfaction = 0.0
            else:
                satisfaction = calculate_user_satisfaction(
                    selected_components_df=selected_components_df,
                    user_needs_df=user_needs_df,
                )

            options.append(
                {
                    "category": category,
                    "component_names": component_names,
                    "cost": cost,
                    "satisfaction": satisfaction,
                    "number_of_components": len(component_names),
                }
            )

        category_options.append(
            {
                "category": category,
                "options": options,
            }
        )

    return category_options

def get_total_combinations(category_options: list[dict]) -> int:
    total = 1

    for category_data in category_options:
        total *= len(category_data["options"])

    return total

def index_to_option_indices(index: int, option_counts: list[int]) -> list[int]:
    option_indices = [0] * len(option_counts)

    for i in range(len(option_counts) - 1, -1, -1):
        option_indices[i] = index % option_counts[i]
        index //= option_counts[i]

    return option_indices

def build_selected_components_from_indices(
    option_indices: list[int],
    category_options: list[dict],
) -> dict[str, list[str]]:
    selected_components = {}

    for category_index, option_index in enumerate(option_indices):
        option = category_options[category_index]["options"][option_index]

        if len(option["component_names"]) > 0:
            selected_components[option["category"]] = option["component_names"]

    return selected_components

def init_worker(
    category_options: list[dict],
    markup: float,
) -> None:
    global _WORKER_OPTIONS
    global _WORKER_OPTION_COUNTS
    global _WORKER_TOTAL_CATEGORIES
    global _WORKER_MARKUP

    _WORKER_OPTIONS = category_options
    _WORKER_OPTION_COUNTS = [
        len(category_data["options"]) for category_data in category_options
    ]
    _WORKER_TOTAL_CATEGORIES = len(category_options)
    _WORKER_MARKUP = markup

def evaluate_index_range(index_range: tuple[int, int]) -> dict:
    assert _WORKER_OPTIONS is not None
    assert _WORKER_OPTION_COUNTS is not None

    start_index, end_index = index_range
    size = end_index - start_index

    costs = np.empty(size, dtype=np.float64)
    satisfactions = np.empty(size, dtype=np.float64)

    best_local_satisfaction = -math.inf
    best_local_index = -1
    best_local_cost = 0.0
    best_local_number_of_components = 0

    for local_position, design_index in enumerate(range(start_index, end_index)):
        option_indices = index_to_option_indices(
            index=design_index,
            option_counts=_WORKER_OPTION_COUNTS,
        )

        cost = 0.0
        satisfaction = 0.0
        number_of_components = 0

        for category_index, option_index in enumerate(option_indices):
            option = _WORKER_OPTIONS[category_index]["options"][option_index]
            cost += option["cost"]
            satisfaction += option["satisfaction"]
            number_of_components += option["number_of_components"]

        costs[local_position] = cost
        satisfactions[local_position] = satisfaction

        if satisfaction > best_local_satisfaction:
            best_local_satisfaction = satisfaction
            best_local_index = design_index
            best_local_cost = cost
            best_local_number_of_components = number_of_components

    return {
        "start_index": start_index,
        "end_index": end_index,
        "costs": costs,
        "satisfactions": satisfactions,
        "best_index": best_local_index,
        "best_cost": best_local_cost,
        "best_satisfaction": best_local_satisfaction,
        "best_number_of_components": best_local_number_of_components,
    }

def load_checkpoint_state() -> dict | None:
    if not CHECKPOINT_STATE_PATH.exists():
        return None

    with open(CHECKPOINT_STATE_PATH, "r") as f:
        return json.load(f)

def save_checkpoint_state(state: dict) -> None:
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    temporary_path = CHECKPOINT_STATE_PATH.with_suffix(".tmp")

    with open(temporary_path, "w") as f:
        json.dump(state, f, indent=4)

    temporary_path.replace(CHECKPOINT_STATE_PATH)

def estimate_demand_parameters(
    user_needs_df: pd.DataFrame,
    components_df: pd.DataFrame,
    categories_df: pd.DataFrame,
    markup: float = 1.5,
    base_market_size: int = 100,
    chunk_size: int = 100_000,
    resume: bool = True,
) -> dict:
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    components_df = components_df.copy()
    categories_df = categories_df.copy()
    user_needs_df = user_needs_df.copy()

    components_df.columns = components_df.columns.str.strip()
    categories_df.columns = categories_df.columns.str.strip()
    user_needs_df.columns = user_needs_df.columns.str.strip()

    components_df["component_category"] = components_df["component_category"].str.strip()
    components_df["component_name"] = components_df["component_name"].str.strip()
    categories_df["categories"] = categories_df["categories"].str.strip()
    user_needs_df["Need"] = user_needs_df["Need"].str.strip()

    if resume and OPTIONS_PATH.exists():
        with open(OPTIONS_PATH, "rb") as f:
            category_options = pickle.load(f)
    else:
        category_options = precompute_category_options(
            components_df=components_df,
            categories_df=categories_df,
            user_needs_df=user_needs_df,
        )

        with open(OPTIONS_PATH, "wb") as f:
            pickle.dump(category_options, f)

    total_combinations = get_total_combinations(category_options)

    state = load_checkpoint_state() if resume else None

    if state is None:
        completed_until = 0
        best_index = -1
        best_cost = 0.0
        best_satisfaction = -math.inf
        best_number_of_components = 0

        costs_array = np.lib.format.open_memmap(
            COST_ARRAY_PATH,
            mode="w+",
            dtype=np.float64,
            shape=(total_combinations,),
        )

        satisfactions_array = np.lib.format.open_memmap(
            SATISFACTION_ARRAY_PATH,
            mode="w+",
            dtype=np.float64,
            shape=(total_combinations,),
        )

    else:
        completed_until = int(state["completed_until"])
        best_index = int(state["best_index"])
        best_cost = float(state["best_cost"])
        best_satisfaction = float(state["best_satisfaction"])
        best_number_of_components = int(state["best_number_of_components"])

        costs_array = np.lib.format.open_memmap(
            COST_ARRAY_PATH,
            mode="r+",
            dtype=np.float64,
            shape=(total_combinations,),
        )

        satisfactions_array = np.lib.format.open_memmap(
            SATISFACTION_ARRAY_PATH,
            mode="r+",
            dtype=np.float64,
            shape=(total_combinations,),
        )

    workers = max(1, (os.cpu_count() or 2) - 1)

    index_ranges = [
        (start, min(start + chunk_size, total_combinations))
        for start in range(completed_until, total_combinations, chunk_size)
    ]

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=init_worker,
        initargs=(category_options, markup),
    ) as executor:
        futures = [
            executor.submit(evaluate_index_range, index_range)
            for index_range in index_ranges
        ]

        with tqdm(
            total=total_combinations,
            initial=completed_until,
            desc=f"Evaluating designs using {workers} cores",
        ) as progress_bar:
            for future in as_completed(futures):
                result = future.result()

                start_index = result["start_index"]
                end_index = result["end_index"]

                costs_array[start_index:end_index] = result["costs"]
                satisfactions_array[start_index:end_index] = result["satisfactions"]

                if result["best_satisfaction"] > best_satisfaction:
                    best_index = int(result["best_index"])
                    best_cost = float(result["best_cost"])
                    best_satisfaction = float(result["best_satisfaction"])
                    best_number_of_components = int(
                        result["best_number_of_components"]
                    )

                completed_until = max(completed_until, end_index)

                save_checkpoint_state(
                    {
                        "completed_until": completed_until,
                        "total_combinations": total_combinations,
                        "best_index": best_index,
                        "best_cost": best_cost,
                        "best_satisfaction": best_satisfaction,
                        "best_number_of_components": best_number_of_components,
                    }
                )

                costs_array.flush()
                satisfactions_array.flush()

                progress_bar.update(end_index - start_index)

    max_satisfaction = float(np.max(satisfactions_array))

    if max_satisfaction == 0:
        raise ValueError(
            "Maximum satisfaction is zero, so satisfaction cannot be normalized."
        )

    normalized_satisfactions = satisfactions_array / max_satisfaction
    selling_prices = costs_array * markup

    reference_satisfaction = float(np.median(normalized_satisfactions))
    reference_price = float(np.median(selling_prices))

    satisfaction_weight = 2.0
    price_sensitivity = 1.0 / reference_price

    best_option_indices = index_to_option_indices(
        index=best_index,
        option_counts=[
            len(category_data["options"]) for category_data in category_options
        ],
    )

    best_selected_components = build_selected_components_from_indices(
        option_indices=best_option_indices,
        category_options=category_options,
    )

    best_normalized_satisfaction = float(best_satisfaction / max_satisfaction)

    return {
        "base_market_size": base_market_size,
        "reference_satisfaction": reference_satisfaction,
        "reference_price": reference_price,
        "max_satisfaction": max_satisfaction,
        "satisfaction_weight": satisfaction_weight,
        "price_sensitivity": price_sensitivity,
        "best_satisfaction_design": {
            "selected_components": best_selected_components,
            "cost": float(best_cost),
            "selling_price": float(best_cost * markup),
            "satisfaction": float(best_satisfaction),
            "normalized_satisfaction": best_normalized_satisfaction,
            "number_of_components": int(best_number_of_components),
        },
        "design_space_summary": {
            "number_of_possible_designs": int(total_combinations),
            "min_cost": float(np.min(costs_array)),
            "median_cost": float(np.median(costs_array)),
            "max_cost": float(np.max(costs_array)),
            "min_selling_price": float(np.min(selling_prices)),
            "median_selling_price": float(np.median(selling_prices)),
            "max_selling_price": float(np.max(selling_prices)),
            "min_satisfaction": float(np.min(satisfactions_array)),
            "median_satisfaction": float(np.median(satisfactions_array)),
            "max_satisfaction": float(np.max(satisfactions_array)),
            "min_normalized_satisfaction": float(np.min(normalized_satisfactions)),
            "median_normalized_satisfaction": float(
                np.median(normalized_satisfactions)
            ),
            "max_normalized_satisfaction": float(np.max(normalized_satisfactions)),
        },
    }


if __name__ == "__main__":

    from load_local import (
        load_categories_df_locally,
        load_components_df_locally,
        load_user_needs_df_locally,
    )

    user_needs_df = load_user_needs_df_locally()
    components_df = load_components_df_locally()
    categories_df = load_categories_df_locally()

    demand_parameters = estimate_demand_parameters(
        user_needs_df=user_needs_df,
        components_df=components_df,
        categories_df=categories_df,
        resume=True,
    )

    print(demand_parameters)
    save_demand_parameters_locally(demand_parameters)