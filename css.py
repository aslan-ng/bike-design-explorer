css = """
.options-grid {
    display: grid !important;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 12px;
    width: 100%;
}

.option-card {
    width: 100%;
    min-height: 90px;
    box-sizing: border-box;
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
    .options-grid {
        grid-template-columns: 1fr;
    }
}
"""