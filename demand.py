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
    
    from load_local import load_demand_parameters_locally

    demand_df = load_demand_parameters_locally()
    print(demand_df)
