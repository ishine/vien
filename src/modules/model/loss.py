import torch


def discriminator_loss(real, fake):
    loss = 0
    for dr, dg in zip(real, fake):
        real_loss = (1 - dr).pow(2).mean()
        fake_loss = dg.pow(2).mean()
        loss += real_loss + fake_loss
    return loss


def generator_loss(disc_outputs):
    loss = 0
    for dg in disc_outputs:
        loss += (1 - dg).pow(2).mean()
    return loss


def fm_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl.detach() - gl))
    return 2 * loss


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p)**2) * torch.exp(-2. * logs_p)
    kl = torch.sum(kl * z_mask)
    l = kl / torch.sum(z_mask)
    return l

