def count_parameters_in_MB(model):
    """Counts trainable parameters in the model in MB."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
