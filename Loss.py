import torch

def Loss(x,y,mu,log_sig, beta):

    l = torch.sum(torch.abs(x.view(-1,28*28)-y.view(-1,28*28)))
    kl = torch.mean(-0.5*torch.sum(1 + log_sig - mu*mu - torch.exp(log_sig), dim=1), dim=0)

    loss = l + beta*kl

    return loss 