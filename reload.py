"""
Complete pipeline to be used when data is updated.
"""

import os
from dotenv import load_dotenv
from huggingface_hub import HfApi

from model import Model
from upload import upload
from load_local import (
    load_categories_df_locally,
    load_components_df_locally,
    load_user_needs_df_locally,
)
from calculate_demand_parameters import (
    estimate_demand_parameters,
    save_demand_parameters_locally,
)
from example import selected_component_dict_0 as selected_component_dict


# Analyse dataset
user_needs_df = load_user_needs_df_locally()
components_df = load_components_df_locally()
categories_df = load_categories_df_locally()

demand_parameters = estimate_demand_parameters(
    user_needs_df=user_needs_df,
    components_df=components_df,
    categories_df=categories_df,
    resume=True,
)

save_demand_parameters_locally(demand_parameters)


# Upload the data to HF
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
api = HfApi(token=HF_TOKEN)
print(api.whoami())
upload(api)


# Check the model working by online data
from load_hf import (
    load_categories_df_from_hf,
    load_components_df_from_hf,
    load_demand_parameters_from_hf,
    load_user_needs_df_from_hf,
)

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