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

    import os
    from dotenv import load_dotenv
    from huggingface_hub import HfApi

    from load_hf import (
        load_categories_df_from_hf,
        load_components_df_from_hf,
        load_demand_parameters_from_hf,
        load_user_needs_df_from_hf,
    )
    from example import selected_component_dict_0 as selected_component_dict

    load_dotenv()

    HF_TOKEN = os.getenv("HF_TOKEN")
    api = HfApi(token=HF_TOKEN)

    model = Model(
        categories_df=load_categories_df_from_hf(cache=False),
        components_df=load_components_df_from_hf(cache=False),
        demand_df=load_demand_parameters_from_hf(cache=False),
        user_needs_df=load_user_needs_df_from_hf(cache=False),
    )

    result = model.evaluate(
        selected_components=selected_component_dict,
        selling_price=4000,
    )

    print(result)