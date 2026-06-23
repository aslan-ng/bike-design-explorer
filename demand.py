def calculate_demand(
    selling_price: float,
    user_satisfaction: float,
    demand_parameters: dict | None = None,
) -> int:
    """
    Calculates demand using calibrated demand parameters.

    Demand increases with normalized satisfaction.
    Demand decreases as selling price moves above the reference price.
    """
    if demand_parameters is None:
        demand_parameters = load_demand_parameters_locally()

    base_market_size = demand_parameters["base_market_size"]
    reference_satisfaction = demand_parameters["reference_satisfaction"]
    reference_price = demand_parameters["reference_price"]
    max_satisfaction = demand_parameters["max_satisfaction"]
    satisfaction_weight = demand_parameters["satisfaction_weight"]
    price_sensitivity = demand_parameters["price_sensitivity"]

    normalized_satisfaction = user_satisfaction / max_satisfaction

    satisfaction_effect = (
        normalized_satisfaction / reference_satisfaction
    ) ** satisfaction_weight

    price_effect = max(
        0.0,
        1.0 - price_sensitivity * (selling_price - reference_price),
    )

    demand = (
        base_market_size
        * satisfaction_effect
        * price_effect
    )

    return max(0, int(round(demand)))

def calculate_profit(
    selling_price: float,
    design_cost: float,
    demand: int,
) -> float:
    return float((selling_price - design_cost) * demand)


if __name__ == "__main__":
    
    from load_local import (
        load_categories_df_locally,
        load_components_df_locally,
        load_demand_parameters_locally,
        load_user_needs_df_locally,
    )
    from design import (
        calculate_design_cost,
        calculate_user_satisfaction,
        select_design_components,
    )
    from example import selected_component_dict_0 as selected_component_dict

    selling_price = 4000

    categories_df = load_categories_df_locally()
    components_df = load_components_df_locally()
    demand_df = load_demand_parameters_locally()
    user_needs_df = load_user_needs_df_locally()

    selected_components_df = select_design_components(
        selected_components=selected_component_dict,
        components_df=components_df,
        categories_df=categories_df,
    )
    user_satisfaction = calculate_user_satisfaction(
        selected_components_df=selected_components_df,
        user_needs_df=user_needs_df,
    )
    design_cost = calculate_design_cost(selected_components_df=selected_components_df)
    demand = calculate_demand(
        selling_price=selling_price,
        user_satisfaction=user_satisfaction,
        demand_parameters=demand_df
    )
    profit = calculate_profit(
        selling_price=selling_price,
        design_cost=design_cost,
        demand=demand
    )
    print("User satisfaction: ", user_satisfaction)
    print("Demand: ", demand)
    print("Profit: ", profit)
