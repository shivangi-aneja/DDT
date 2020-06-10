import torch
from config import orig_weight_factor, div_loss, mean1, mean2


# Divergence loss summed over all elements and batch normalized
def ddt_loss(mu, logvar, labels):
    mu = torch.abs(mu)

    orig_indices = (labels == 0).nonzero().squeeze(1)
    fake_indices = (labels == 1).nonzero().squeeze(1)

    mu_orig = torch.index_select(mu, dim=0, index=orig_indices, out=None)
    mu_fake = torch.index_select(mu, dim=0, index=fake_indices, out=None)

    logvar_orig = torch.index_select(logvar, dim=0, index=orig_indices, out=None)
    logvar_fake = torch.index_select(logvar, dim=0, index=fake_indices, out=None)

    kl_orig = orig_weight_factor * div_loss(mu=mu_orig, logvar=logvar_orig, mean=mean1)
    kl_fake = div_loss(mu=mu_fake, logvar=logvar_fake, mean=mean2)

    real_count = mu_orig.shape[0]
    fake_count = mu_fake.shape[0]
    kl_loss = (kl_orig + kl_fake) / (fake_count + orig_weight_factor * real_count)

    loss = kl_loss
    return loss