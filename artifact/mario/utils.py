import torch

def kl_normal(mu, logvar, reduction: str = "mean"):
    # KL = -0.5 * (1 + logvar - mu^2 - exp(logvar))
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

    if reduction == "mean":
        return kl.mean()
    elif reduction == "sum":
        return kl.sum()
    elif reduction == "none":
        return kl
    else:
        raise ValueError(f"Invalid reduction: {reduction}")