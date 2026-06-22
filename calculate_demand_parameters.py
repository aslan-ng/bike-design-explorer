from itertools import product, combinations
import json
from pathlib import Path
import pandas as pd

from design import calculate_design_cost, calculate_design_satisfaction, select_design_components
from components import load_components_df_locally
from categories import load_categories_df_locally
from user_needs import load_user_needs_df_locally
from demand import DEMAND_FILE_PATH


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


def generate_all_design_selections(
    components_df: pd.DataFrame,
    categories_df: pd.DataFrame,
) -> list[dict[str, list[str]]]:
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

    all_designs = []

    categories = [category for category, _ in category_options]
    options_by_category = [options for _, options in category_options]

    for design_options in product(*options_by_category):
        selected_components = {
            category: component_names
            for category, component_names in zip(categories, design_options)
            if len(component_names) > 0
        }

        all_designs.append(selected_components)

    return all_designs


def estimate_demand_parameters(
    user_needs_df: pd.DataFrame,
    components_df: pd.DataFrame,
    categories_df: pd.DataFrame,
    markup: float = 1.5,
    base_market_size: int = 100,
) -> dict:
    """
    Estimates demand-function parameters from the available design space.

    New schema:
    - categories.csv defines required categories
    - categories.csv defines whether multiple selections are allowed
    - components are selected by category + component_name
    """

    components_df.columns = components_df.columns.str.strip()
    user_needs_df.columns = user_needs_df.columns.str.strip()
    user_needs_df["Need"] = user_needs_df["Need"].str.strip()

    all_design_selections = generate_all_design_selections(
        components_df=components_df,
        categories_df=categories_df,
    )

    all_designs = []

    for selected_components_dict in all_design_selections:
        selected_components_df = select_design_components(
            selected_components=selected_components_dict,
            components_df=components_df,
            categories_df=categories_df,
        )

        cost = calculate_design_cost(selected_components_df)

        satisfaction = calculate_design_satisfaction(
            selected_components_df=selected_components_df,
            user_needs_df=user_needs_df,
        )

        all_designs.append(
            {
                "selected_components": selected_components_dict,
                "cost": cost,
                "selling_price": cost * markup,
                "satisfaction": satisfaction,
                "number_of_components": len(selected_components_df),
            }
        )

    design_space_df = pd.DataFrame(all_designs)

    best_satisfaction_position = int(design_space_df["satisfaction"].to_numpy().argmax())
    best_satisfaction_design = design_space_df.iloc[best_satisfaction_position]

    max_satisfaction = float(best_satisfaction_design["satisfaction"])

    if max_satisfaction == 0:
        raise ValueError("Maximum satisfaction is zero, so satisfaction cannot be normalized.")

    design_space_df["normalized_satisfaction"] = (
        design_space_df["satisfaction"] / max_satisfaction
    )

    reference_satisfaction = float(
        design_space_df["normalized_satisfaction"].median()
    )

    reference_price = float(
        design_space_df["selling_price"].median()
    )

    # Demand idea:
    # satisfaction helps demand, but price hurts demand.
    # Therefore, the highest-satisfaction design is not automatically the best business design.
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
            "normalized_satisfaction": float(best_satisfaction_design["normalized_satisfaction"]),
            "number_of_components": int(best_satisfaction_design["number_of_components"]),
        },
        "design_space_summary": {
            "number_of_possible_designs": len(design_space_df),
            "min_cost": float(design_space_df["cost"].min()),
            "median_cost": float(design_space_df["cost"].median()),
            "max_cost": float(design_space_df["cost"].max()),
            "min_selling_price": float(design_space_df["selling_price"].min()),
            "median_selling_price": float(design_space_df["selling_price"].median()),
            "max_selling_price": float(design_space_df["selling_price"].max()),
            "min_satisfaction": float(design_space_df["satisfaction"].min()),
            "median_satisfaction": float(design_space_df["satisfaction"].median()),
            "max_satisfaction": float(design_space_df["satisfaction"].max()),
            "min_normalized_satisfaction": float(design_space_df["normalized_satisfaction"].min()),
            "median_normalized_satisfaction": float(design_space_df["normalized_satisfaction"].median()),
            "max_normalized_satisfaction": float(design_space_df["normalized_satisfaction"].max()),
        },
    }


if __name__ == "__main__":
    user_needs_df = load_user_needs_df_locally()
    components_df = load_components_df_locally()
    categories_df = load_categories_df_locally()

    demand_parameters = estimate_demand_parameters(
        user_needs_df=user_needs_df,
        components_df=components_df,
        categories_df=categories_df,
    )

    save_demand_parameters_locally(demand_parameters)