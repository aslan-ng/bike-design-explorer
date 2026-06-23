import os

import gradio as gr
from huggingface_hub import HfApi

from model import Model
from load_hf import (
    load_categories_df_from_hf,
    load_components_df_from_hf,
    load_demand_parameters_from_hf,
    load_user_needs_df_from_hf,
)


HF_TOKEN = os.getenv("HF_TOKEN")
api = HfApi(token=HF_TOKEN)

categories_df = load_categories_df_from_hf(cache=True)
components_df = load_components_df_from_hf(cache=True)
demand_df = load_demand_parameters_from_hf(cache=True)
user_needs_df = load_user_needs_df_from_hf(cache=True)

model = Model(
    categories_df=categories_df,
    components_df=components_df,
    demand_df=demand_df,
    user_needs_df=user_needs_df,
)


css = """
.options-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
}
.option-card {
    flex: 1 1 260px;
    min-width: 220px;
    border: 1px solid #4b4b55;
    border-radius: 8px;
    padding: 10px 14px;
    cursor: pointer;
    background: #25252a;
}
.option-card:hover {
    border-color: #ff7a1a;
}
.option-title {
    font-size: 1rem;
    font-weight: 700;
}
.option-description {
    font-size: 0.85rem;
    font-weight: 400;
    color: #b8b8b8;
    margin-top: 4px;
    line-height: 1.25;
}
.selected-card {
    border-color: #ff7a1a;
    box-shadow: 0 0 0 2px #ff7a1a;
}
@media (max-width: 640px) {
    .option-card {
        flex-basis: 100%;
    }
}
"""


def make_card_html(
    name: str,
    cost: float,
    description: str,
    selected_names: list[str],
) -> str:
    selected_class = " selected-card" if name in selected_names else ""
    return f"""
    <div class="option-card{selected_class}">
        <div class="option-title">{name} (${cost})</div>
        <div class="option-description">{description}</div>
    </div>
    """


def make_select_component_fn(
    options: list[dict],
    component_name: str,
    allow_multiple: bool,
):
    def select_component(selected_names):
        selected_names = selected_names or []

        if allow_multiple:
            if component_name in selected_names:
                selected_names = [
                    name for name in selected_names
                    if name != component_name
                ]
            else:
                selected_names = [*selected_names, component_name]
        else:
            selected_names = [component_name]

        updated_cards = [
            make_card_html(
                name=option["component_name"],
                cost=option["component_cost"],
                description=option["component_description"],
                selected_names=selected_names,
            )
            for option in options
        ]

        return [selected_names, *updated_cards]

    return select_component


category_states = []


def get_category_settings(category: str) -> tuple[bool, bool]:
    rows = categories_df[categories_df["categories"] == category]

    if rows.empty:
        return True, False

    row = rows.iloc[0]

    required = bool(row["required"])
    allow_multiple = bool(row["multiple_allowed"])

    return required, allow_multiple


def evaluate_ui(selling_price, *category_values):
    if selling_price is None or selling_price <= 0:
        return "Please enter a valid positive selling price."

    selected_components = {}

    for category_info, selected_names in zip(category_states, category_values):
        category = category_info["category"]
        required = category_info["required"]

        selected_names = selected_names or []
        selected_components[category] = selected_names

        if required and len(selected_names) == 0:
            return f"Please choose one item from required category: {category}"

    result = model.evaluate(
        selected_components=selected_components,
        selling_price=selling_price,
    )

    return f"""
### Evaluation Results

| Metric | Value |
|---|---:|
| Design Cost | ${result["design_cost"]:,.2f} |
| User Satisfaction | {result["user_satisfaction"]:,.0f} |
| Demand | {result["demand"]:,} |
| Profit | ${result["profit"]:,.2f} |
"""


with gr.Blocks(title="Bike Design Explorer") as demo:
    gr.Markdown(
        """
        # Bike Design Explorer

        Choose bike components, then set a unit selling price to evaluate
        cost, user satisfaction, demand, and profit.
        """
    )

    category_state_inputs = []

    for category in sorted(components_df["component_category"].unique()):
        category_df = components_df[
            components_df["component_category"] == category
        ]

        required, allow_multiple = get_category_settings(category)

        label_suffix = []
        label_suffix.append("required" if required else "optional")
        label_suffix.append("multiple allowed" if allow_multiple else "choose one")

        gr.Markdown(f"## {category} ({', '.join(label_suffix)})")

        options = category_df.to_dict("records")

        selected_state = gr.State(value=[])
        category_state_inputs.append(selected_state)

        category_states.append(
            {
                "category": category,
                "required": required,
                "allow_multiple": allow_multiple,
            }
        )

        card_outputs = []

        with gr.Group(elem_classes=["options-grid"]):
            for option in options:
                card = gr.HTML(
                    make_card_html(
                        name=option["component_name"],
                        cost=option["component_cost"],
                        description=option["component_description"],
                        selected_names=[],
                    )
                )
                card_outputs.append(card)

        for option, card in zip(options, card_outputs):
            card.click(
                fn=make_select_component_fn(
                    options=options,
                    component_name=option["component_name"],
                    allow_multiple=allow_multiple,
                ),
                inputs=[selected_state],
                outputs=[selected_state, *card_outputs],
            )

    gr.Markdown("## Price")

    selling_price = gr.Number(
        label="Unit Selling Price ($)",
        value=None,
        minimum=0,
    )

    evaluate_button = gr.Button("Evaluate Design")
    output = gr.Markdown(label="Evaluation Results")

    evaluate_button.click(
        fn=evaluate_ui,
        inputs=[selling_price, *category_state_inputs],
        outputs=output,
    )


if __name__ == "__main__":
    demo.queue()
    demo.launch(css=css)