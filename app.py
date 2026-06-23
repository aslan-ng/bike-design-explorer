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
import gradio as gr

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


component_grouping_states = []


def evaluate_ui(selling_price, *group_values):
    if selling_price is None or selling_price <= 0:
        return "Please enter a valid positive selling price."

    selected_components = {}

    for grouping_info, selected_names in zip(
        component_grouping_states,
        group_values,
    ):
        grouping = grouping_info["grouping"]
        required = grouping_info["required"]

        selected_names = selected_names or []
        selected_components[grouping] = selected_names

        if required and len(selected_names) == 0:
            return f"Please choose one item from required group: {grouping}"

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


def get_grouping_settings(grouping: str) -> tuple[bool, bool]:
    grouping_rows = categories_df[
        categories_df["component_grouping"] == grouping
    ]

    if grouping_rows.empty:
        return True, False

    row = grouping_rows.iloc[0]

    required = bool(row.get("required", True))
    allow_multiple = bool(row.get("allow_multiple", False))

    return required, allow_multiple


with gr.Blocks(
    title="Bike Design Explorer",
    css=css,
) as demo:
    gr.Markdown(
        """
        # Bike Design Explorer

        Choose bike components, then set a unit selling price to evaluate
        cost, user satisfaction, demand, and profit.
        """
    )

    group_state_inputs = []

    for category in sorted(components_df["component_category"].unique()):
        category_df = components_df[
            components_df["component_category"] == category
        ]

        gr.Markdown(f"## {category}")

        with gr.Group():
            for grouping in sorted(category_df["component_grouping"].unique()):
                grouping_df = category_df[
                    category_df["component_grouping"] == grouping
                ]

                required, allow_multiple = get_grouping_settings(grouping)

                label_suffix = []
                if required:
                    label_suffix.append("required")
                else:
                    label_suffix.append("optional")

                if allow_multiple:
                    label_suffix.append("multiple allowed")
                else:
                    label_suffix.append("choose one")

                gr.Markdown(
                    f"### {grouping} "
                    f"({', '.join(label_suffix)})"
                )

                options = grouping_df.to_dict("records")

                selected_state = gr.State(value=[])
                group_state_inputs.append(selected_state)

                component_grouping_states.append(
                    {
                        "grouping": grouping,
                        "required": required,
                        "allow_multiple": allow_multiple,
                    }
                )

                card_outputs = []

                with gr.Row(elem_classes=["options-grid"]):
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
        inputs=[selling_price, *group_state_inputs],
        outputs=output,
    )


if __name__ == "__main__":
    demo.queue()
    demo.launch()