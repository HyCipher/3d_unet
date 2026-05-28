import torch


def build_optimizer(model, controls, lr):
    optimizer_name = str(controls.get("optimizer", "adam")).lower()
    weight_decay = float(controls.get("weight_decay", 0.0))

    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer_name == "sgd":
        momentum = float(controls.get("momentum", 0.9))
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )

    raise ValueError(
        f"Unsupported optimizer: {optimizer_name}. "
        "Use one of: adam, adamw, sgd."
    )