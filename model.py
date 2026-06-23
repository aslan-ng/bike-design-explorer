import pandas as pd

from design import (
    select_design_components,
    calculate_design_cost,
    calculate_user_satisfaction,
)
from demand import (
    calculate_profit,
    calculate_demand,
)


class Model:

    def __init__(
        self,
        categories_df: pd.DataFrame,
        components_df: pd.DataFrame,
        demand_df: dict,
        user_needs_df: pd.DataFrame,
    ):
        self.categories_df = categories_df
        self.components_df = components_df
        self.demand_df = demand_df
        self.user_needs_df = user_needs_df

    def evaluate(
        self,
        selected_components: dict[str, list[str]],
        selling_price: float,
    ) -> dict:
        selected_components_df = select_design_components(
            selected_components=selected_components,
            components_df=self.components_df,
            categories_df=self.categories_df,
        )

        design_cost = calculate_design_cost(
            selected_components_df=selected_components_df,
        )

        user_satisfaction = calculate_user_satisfaction(
            selected_components_df=selected_components_df,
            user_needs_df=self.user_needs_df,
        )

        demand = calculate_demand(
            selling_price=selling_price,
            user_satisfaction=user_satisfaction,
            demand_parameters=self.demand_df,
        )

        profit = calculate_profit(
            selling_price=selling_price,
            design_cost=design_cost,
            demand=demand,
        )

        return {
            "design_cost": design_cost,
            "selling_price": selling_price,
            "user_satisfaction": user_satisfaction,
            "demand": demand,
            "profit": profit,
        }


if __name__ == "__main__":

    from load_local import (
        load_categories_df_locally,
        load_components_df_locally,
        load_demand_parameters_locally,
        load_user_needs_df_locally,
    )
    from example import selected_component_dict_0 as selected_component_dict

    model = Model(
        categories_df=load_categories_df_locally(),
        components_df=load_components_df_locally(),
        demand_df=load_demand_parameters_locally(),
        user_needs_df=load_user_needs_df_locally(),
    )

    result = model.evaluate(
        selected_components=selected_component_dict,
        selling_price=4000,
    )

    print(result)