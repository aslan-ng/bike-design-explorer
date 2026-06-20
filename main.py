import pandas as pd

from data_loader import *
from utils import *
from demand import *


class Model:
    
    def __init__(
        self,
        customer_research_df: pd.DataFrame,
        components_df: pd.DataFrame,
    ):
        self.customer_research_df = customer_research_df
        self.components_df = components_df
    
    def evaluate(
        self,
        selected_component_names: list,
        selling_price: float,
    ):
        selected_components = select_design_components(
            selected_component_names=selected_component_names,
            components_df=self.components_df
        )
        design_cost = calculate_design_cost(selected_components)
        satisfaction = calculate_design_satisfaction(
            selected_components=selected_components,
            customer_research_df=self.customer_research_df,
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
    # Load data
    customer_research_df = load_customer_research_df_locally(DEFAULT_CUSTOMER_RESEARCH_FILE)
    components_df = load_components_df_locally(DEFAULT_COMPONENTS_FILE)

    m = Model(
        components_df=components_df,
        customer_research_df=customer_research_df
    )
    print(m.components)
    print(m.evaluate
        (
            selected_component_names=['Aluminum Frame', 'Standard Tire'],
            selling_price=1000
        )
    )
    print(m.evaluate
        (
            selected_component_names=['Carbon Frame', 'Standard Tire'],
            selling_price=1000
        )
    )
    print(m.evaluate
        (
            selected_component_names=['Carbon Frame', 'Wide Tire'],
            selling_price=1000
        )
    )