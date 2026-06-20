import pandas as pd

from data_loader import *


def select_design_components(
    selected_component_names: list[str],
    components_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Selects components by component_name.

    Rules:
    - exactly one component per grouping
    - every grouping must be represented

    Categories are descriptive only.
    Groupings define mutually exclusive selection slots.
    """
    selected_df = components_df[
        components_df["component_name"].isin(selected_component_names)
    ].copy()
    if len(selected_df) != len(selected_component_names):
        found = set(selected_df["component_name"])
        requested = set(selected_component_names)
        missing = requested - found
        raise ValueError(
            f"Some selected components were not found: {missing}"
        )
    required_groupings = set(get_groupings(components_df))
    selected_groupings = selected_df["component_grouping"].tolist()
    duplicate_groupings = {
        grouping
        for grouping in selected_groupings
        if selected_groupings.count(grouping) > 1
    }
    if duplicate_groupings:
        raise ValueError(
            f"Only one component per grouping is allowed. "
            f"Duplicate groupings: {duplicate_groupings}"
        )
    missing_groupings = required_groupings - set(selected_groupings)
    if missing_groupings:
        raise ValueError(
            f"Every grouping must have one selected component. "
            f"Missing groupings: {missing_groupings}"
        )
    return selected_df

def calculate_design_cost(selected_components: pd.DataFrame) -> float:
    return float(selected_components["component_cost"].sum())

def calculate_design_satisfaction(
    selected_components: pd.DataFrame,
    customer_research_df: pd.DataFrame,
) -> float:
    """
    Calculates satisfaction.
    """
    customer_importance = get_customer_importance(customer_research_df)
    satisfaction = 0.0
    for feature, importance in customer_importance.items():
        component_contributions = selected_components[feature]
        total_feature_contribution = component_contributions.sum()
        satisfaction += importance * total_feature_contribution
    return float(satisfaction)


if __name__ == "__main__":
    # Load data
    customer_research_df = load_customer_research_df_locally(DEFAULT_CUSTOMER_RESEARCH_FILE)
    components_df = load_components_df_locally(DEFAULT_COMPONENTS_FILE)

    selected_components = select_design_components(
        selected_component_names=['Aluminum Frame', 'Standard Tire'],
        components_df=components_df
    )
    #print(selected_components)

    design_cost = calculate_design_cost(selected_components)
    #print(design_cost)

    satisfaction = calculate_design_satisfaction(
        selected_components=selected_components,
        customer_research_df=customer_research_df,
    )
    #print(satisfaction)
    



