import pandas as pd


def select_design_components(
    selected_components: dict[str, list[str]],
    components_df: pd.DataFrame,
    categories_df: pd.DataFrame,
) -> pd.DataFrame:

    selected_rows = []

    category_rules = categories_df.set_index("categories")

    for category, component_names in selected_components.items():
        if category not in category_rules.index:
            raise ValueError(f"Unknown category: {category}")

        multiple_allowed = bool(category_rules.loc[category, "multiple_allowed"])

        if len(component_names) > 1 and not multiple_allowed:
            raise ValueError(
                f"Category '{category}' allows only one selected component."
            )

        for component_name in component_names:
            matches = components_df[
                (components_df["component_category"] == category)
                & (components_df["component_name"] == component_name)
            ]

            if len(matches) == 0:
                raise ValueError(
                    f"Component '{component_name}' was not found "
                    f"in category '{category}'."
                )

            if len(matches) > 1:
                raise ValueError(
                    f"Component '{component_name}' appears more than once "
                    f"in category '{category}'."
                )

            selected_rows.append(matches.iloc[0])

    selected_df = pd.DataFrame(selected_rows)

    required_categories = set(
        categories_df.loc[categories_df["required"], "categories"]
    )

    selected_category_set = set(selected_components.keys())
    missing_required_categories = required_categories - selected_category_set

    if missing_required_categories:
        raise ValueError(
            f"Every required category must have one selected component. "
            f"Missing categories: {missing_required_categories}"
        )

    return selected_df.reset_index(drop=True)

def calculate_design_cost(selected_components_df: pd.DataFrame) -> float:
    return float(selected_components_df["component_cost"].sum())

def calculate_user_satisfaction(
    selected_components_df: pd.DataFrame,
    user_needs_df: pd.DataFrame,
) -> float:
    satisfaction = 0.0
    for _, row in user_needs_df.iterrows():
        need = row["Need"]
        importance = row["Importance"]
        total_contribution = (selected_components_df[need]).sum()
        satisfaction += importance * total_contribution
    return float(satisfaction)


if __name__ == "__main__":
    # Load data
    from load_local import (
        load_categories_df_locally,
        load_components_df_locally,
        load_user_needs_df_locally,
    )
    
    categories_df = load_categories_df_locally()
    components_df = load_components_df_locally()
    user_needs_df = load_user_needs_df_locally()
    
    selected_component_dict = {
        'Bike Frame': ['Titanium'],
        'Tires': ['Basic'],
        'Brakes': ['Precision'],
        'Handlebars': ['Comfort Straight'],
        'Pedals': ['Extra Grip Flat'],
        'Gears': ['14 Speed'],
        'Seat': ['Polymer Gel All-Purpose'],
        'Accessories': ['Light'],
    }

    selected_components_df = select_design_components(
        selected_components=selected_component_dict,
        components_df=components_df,
        categories_df=categories_df,
    )
    print(">>> Selected Components:\n", selected_components_df)

    design_cost = calculate_design_cost(
        selected_components_df=selected_components_df,
    )
    print("\n>>> Design Cost: ", design_cost)

    user_satisfaction = calculate_user_satisfaction(
        selected_components_df=selected_components_df,
        user_needs_df=user_needs_df,
    )
    print("\n>>> User Satisfaction: ", user_satisfaction)



