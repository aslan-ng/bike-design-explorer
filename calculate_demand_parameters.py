"""
Analyze the configurations, find the best combination among 55987200 possible configurations.
Create "demand_parameters.json" file.
"""

import os
import json
import pandas as pd
from pathlib import Path
from itertools import product, combinations
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from design import (
    calculate_design_cost,
    calculate_design_satisfaction,
    select_design_components,
)
from demand import DEMAND_FILE_PATH


_WORKER_COMPONENTS_DF = None
_WORKER_CATEGORIES_DF = None
_WORKER_USER_NEEDS_DF = None
_WORKER_MARKUP = None


def save_demand_parameters_locally(
    demand_parameters: dict,
    file_path: str | Path = DEMAND_FILE_PATH,
) -> None:
    with open(file_path, "w") as f:
        json.dump(demand_parameters, f, indent=4)


def init_worker(
    components_df: pd.DataFrame,
    categories_df: pd.DataFrame,
    user_needs_df: pd.DataFrame,
    markup: float,
) -> None:
    global _WORKER_COMPONENTS_DF
    global _WORKER_CATEGORIES_DF
    global _WORKER_USER_NEEDS_DF
    global _WORKER_MARKUP

    _WORKER_COMPONENTS_DF = components_df
    _WORKER_CATEGORIES_DF = categories_df
    _WORKER_USER_NEEDS_DF = user_needs_df
    _WORKER_MARKUP = markup


def evaluate_one_design(selected_components_dict: dict[str, list[str]]) -> dict:
    assert _WORKER_COMPONENTS_DF is not None
    assert _WORKER_CATEGORIES_DF is not None
    assert _WORKER_USER_NEEDS_DF is not None
    assert _WORKER_MARKUP is not None

    selected_components_df = select_design_components(
        selected_components=selected_components_dict,
        components_df=_WORKER_COMPONENTS_DF,
        categories_df=_WORKER_CATEGORIES_DF,
    )

    cost = calculate_design_cost(selected_components_df)

    satisfaction = calculate_design_satisfaction(
        selected_components_df=selected_components_df,
        user_needs_df=_WORKER_USER_NEEDS_DF,
    )

    return {
        "selected_components": selected_components_dict,
        "cost": cost,
        "selling_price": cost * _WORKER_MARKUP,
        "satisfaction": satisfaction,
        "number_of_components": len(selected_components_df),
    }


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


def get_design_selection_iterator_and_total(
    components_df: pd.DataFrame,
    categories_df: pd.DataFrame,
):
    category_options = []

    for _, row in categories_df.iterrows():
        category = row["categories"]
        required = bool(row["required"])
        multiple_allowed = bool(row["multiple_allowed"])

        options = get_category_selection_options(
            category=category,
            components_df=components_df,
            required=required,
            multiple_allowed=multiple_allowed,
        )

        category_options.append((category, options))

    categories = [category for category, _ in category_options]
    options_by_category = [options for _, options in category_options]

    total_combinations = 1
    for options in options_by_category:
        total_combinations *= len(options)

    iterator = (
        {
            category: component_names
            for category, component_names in zip(categories, design_options)
            if len(component_names) > 0
        }
        for design_options in product(*options_by_category)
    )

    return iterator, total_combinations


def estimate_demand_parameters(
    user_needs_df: pd.DataFrame,
    components_df: pd.DataFrame,
    categories_df: pd.DataFrame,
    markup: float = 1.5,
    base_market_size: int = 100,
) -> dict:
    components_df.columns = components_df.columns.str.strip()
    categories_df.columns = categories_df.columns.str.strip()
    user_needs_df.columns = user_needs_df.columns.str.strip()
    user_needs_df["Need"] = user_needs_df["Need"].str.strip()

    design_iterator, total_combinations = get_design_selection_iterator_and_total(
        components_df=components_df,
        categories_df=categories_df,
    )

    workers = max(1, (os.cpu_count() or 2) - 1)

    all_designs = []

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=init_worker,
        initargs=(components_df, categories_df, user_needs_df, markup),
    ) as executor:
        results = executor.map(
            evaluate_one_design,
            design_iterator,
            chunksize=500,
        )

        for result in tqdm(
            results,
            total=total_combinations,
            desc=f"Evaluating designs using {workers} cores",
        ):
            all_designs.append(result)

    design_space_df = pd.DataFrame(all_designs)

    best_satisfaction_position = int(
        design_space_df["satisfaction"].to_numpy().argmax()
    )
    best_satisfaction_design = design_space_df.iloc[best_satisfaction_position]

    max_satisfaction = float(best_satisfaction_design["satisfaction"])

    if max_satisfaction == 0:
        raise ValueError(
            "Maximum satisfaction is zero, so satisfaction cannot be normalized."
        )

    design_space_df["normalized_satisfaction"] = (
        design_space_df["satisfaction"] / max_satisfaction
    )

    reference_satisfaction = float(
        design_space_df["normalized_satisfaction"].median()
    )

    reference_price = float(
        design_space_df["selling_price"].median()
    )

    satisfaction_weight = 2.0
    price_sensitivity = 1.0 / reference_price

    return {
        "base_market_size": base_market_size,
        "reference_satisfaction": reference_satisfaction,
        "reference_price": reference_price,
        "max_satisfaction": max_satisfaction,
        "satisfaction_weight": satisfaction_weight,
        "price_sensitivity": price_sensitivity,
        "best_satisfaction_design": {
            "selected_components": best_satisfaction_design["selected_components"],
            "cost": float(best_satisfaction_design["cost"]),
            "selling_price": float(best_satisfaction_design["selling_price"]),
            "satisfaction": float(best_satisfaction_design["satisfaction"]),
            "normalized_satisfaction": float(
                best_satisfaction_design["normalized_satisfaction"]
            ),
            "number_of_components": int(
                best_satisfaction_design["number_of_components"]
            ),
        },
        "design_space_summary": {
            "number_of_possible_designs": len(design_space_df),
            "min_cost": float(design_space_df["cost"].min()),
            "median_cost": float(design_space_df["cost"].median()),
            "max_cost": float(design_space_df["cost"].max()),
            "min_selling_price": float(design_space_df["selling_price"].min()),
            "median_selling_price": float(
                design_space_df["selling_price"].median()
            ),
            "max_selling_price": float(design_space_df["selling_price"].max()),
            "min_satisfaction": float(design_space_df["satisfaction"].min()),
            "median_satisfaction": float(
                design_space_df["satisfaction"].median()
            ),
            "max_satisfaction": float(design_space_df["satisfaction"].max()),
            "min_normalized_satisfaction": float(
                design_space_df["normalized_satisfaction"].min()
            ),
            "median_normalized_satisfaction": float(
                design_space_df["normalized_satisfaction"].median()
            ),
            "max_normalized_satisfaction": float(
                design_space_df["normalized_satisfaction"].max()
            ),
        },
    }


if __name__ == "__main__":

    from components import load_components_df_locally
    from categories import load_categories_df_locally
    from user_needs import load_user_needs_df_locally

    user_needs_df = load_user_needs_df_locally()
    components_df = load_components_df_locally()
    categories_df = load_categories_df_locally()

    demand_parameters = estimate_demand_parameters(
        user_needs_df=user_needs_df,
        components_df=components_df,
        categories_df=categories_df,
    )

    print(demand_parameters)
    save_demand_parameters_locally(demand_parameters)