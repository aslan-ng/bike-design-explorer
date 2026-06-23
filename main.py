import pandas as pd

from categories import (
    load_categories_df_locally,
    load_categories_df_from_hf,
)
from components import (
    load_components_df_locally,
    load_components_df_from_hf,
)
from demand import (
    load_demand_parameters_locally,
    load_demand_parameters_from_hf,
)
from user_needs import (
    load_user_needs_df_locally,
    load_user_needs_df_from_hf,
)
from design import (
    select_design_components,
    calculate_design_cost,
    calculate_design_satisfaction,
)


class Model:
    
    def __init__(
        self,
        data_source: str = "online",
    ):
        self.load(source=data_source)

    def load(self, source: str):
        if source.lower() in ["local"]:
            self.categories_df = load_categories_df_locally()
            self.components_df = load_components_df_locally()
            self.demand_parameters = load_demand_parameters_locally()
            self.user_needs_df = load_user_needs_df_locally()
        elif source.lower() in ["hf", "online", "repo", "huggingface"]:
            self.categories_df = load_categories_df_from_hf()
            self.components_df = load_components_df_from_hf()
            self.demand_parameters = load_demand_parameters_from_hf()
            self.user_needs_df = load_user_needs_df_from_hf()
        else:
            raise ValueError("Data loading failed.")
    
    def evaluate(
        self,
        selected_components: dict[str, list[str]],
        selling_price: float,
    ):
        selected_components_df = select_design_components(
            selected_components=selected_components,
            components_df=self.components_df,
            categories_df=self.user_needs_df,
        )
        design_cost = calculate_design_cost(selected_components_df)
        satisfaction = calculate_design_satisfaction(
            selected_components_df=selected_components_df,
            user_needs_df=self.user_needs_df,
        )
        demand = calculate_demand(
            selling_price=selling_price,
            satisfaction=satisfaction,
        )
        profit = calculate_profit(
            selling_price=selling_price,
            design_cost=design_cost,
            demand=demand,
        )
        return {
            "design_cost": design_cost,
            "selling_price": selling_price,
            "satisfaction": satisfaction,
            "demand": demand,
            "profit": profit,
        }
    
    @property
    def components(self) -> list:
        return get_components_name(self.components_df)
    

if __name__ == '__main__':
    # Load data locally
    m = Model(data_source="local")

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
    print(m.components)
    print(m.evaluate
        (
            selected_component=selected_component_dict,
            selling_price=1000
        )
    )