import torch
import torch.distributions as tdist
import torch.nn as nn
import math

log_sigmoid = nn.LogSigmoid()
PI = math.pi
EPSILON = torch.tensor(10e-25).double()

class Model(nn.Module):
    def __init__(self, input_size, latent_size, model_name):
        super(Model, self).__init__()

        self.input_size = input_size
        self.latent_size = latent_size
        self.model_name = model_name

        self.encoder_model = nn.Sequential(nn.Linear(input_size+1, 100), nn.Tanh(), nn.Linear(100, latent_size*2))
        if model_name == 'VFAE':
            self.encoder_model_z = nn.Sequential(nn.Linear(latent_size+1, 100), nn.Tanh(), nn.Linear(100, latent_size*2))
            self.classifier = nn.Sequential(nn.Linear(latent_size, 1))
            self.reconst_model_z = nn.Sequential(nn.Linear(latent_size+1, 100), nn.Tanh(), nn.Linear(100, latent_size*2))
            self.decoder = nn.Sequential(nn.Linear(latent_size+1, 100), nn.Tanh(), nn.Linear(100, input_size))
        elif model_name == 'VFIB':
            self.classifier = nn.Sequential(nn.Linear(latent_size + 1, 1))
        elif model_name == 'LCFR': #Unsupervised version of VFAE
            self.decoder = nn.Sequential(nn.Linear(latent_size+1, 100), nn.Tanh(), nn.Linear(100, input_size))
        elif model_name == 'VAE':
            self.decoder = nn.Sequential(nn.Linear(latent_size, 100), nn.Tanh(), nn.Linear(100, input_size))
        elif model_name == 'VFIBG':
            self.classifier = nn.Sequential(nn.Linear(latent_size + 1, 1))
            self.decoder = nn.Sequential(nn.Linear(latent_size + 1, 200), nn.Tanh(), nn.Linear(200, input_size))

    def encoder(self, input):
        out = self.encoder_model(input).view(-1, 2, self.latent_size)
        mu, log_sigma = out[:, 0, :], out[:, 1, :]
        return mu, log_sigma

    def encoder_z(self, input):
        out = self.encoder_model_z(input).view(-1, 2, self.latent_size)
        mu, log_sigma = out[:, 0, :], out[:, 1, :]
        return mu, log_sigma

    def reconst_z(self, input):
        out = self.reconst_model_z(input).view(-1, 2, self.latent_size)
        mu, log_sigma = out[:, 0, :], out[:, 1, :]
        return mu, log_sigma


def reparam(mu, log_sigma):
    std = log_sigma.mul(0.5).exp_()
    return mu + std*torch.randn_like(mu)


def KL(mu, log_sigma):
    return 0.5*(-log_sigma + mu**2 + log_sigma.exp()).mean()


# def negative_log_bernoulli(label, pred, mean=True):
#
#     log_likelihood = -torch.sum(label*pred + (1-label)*(1-pred), dim=1)
#     if mean is False:
#         return torch.squeeze(log_likelihood)
#     return torch.mean(log_likelihood)

def negative_log_bernoulli(data, mu, mean=True, clamp=True):
    if clamp:
        mu = torch.clamp(mu, min=-9.5, max=9.5)
    mdata = data.view(data.shape[0], -1)
    mmu = mu.view(data.shape[0], -1)

    log_prob_1 = log_sigmoid(mmu)
    log_prob_2 = log_sigmoid(-mmu)
    log_likelihood = -torch.mean((mdata*log_prob_1)+(1-mdata)*log_prob_2)
    return log_likelihood

#TODO remove clamp function
def entropy_gaussian(mu, sigma, mean=True):
    msigma = sigma.view(sigma.shape[0], -1)
    return torch.mean(0.5*(msigma))


def negative_log_gaussian(data, mu, sigma, mean=True):
    mdata = data.view(data.shape[0], -1)
    mmu = mu.view(data.shape[0], -1)
    msigma = sigma.view(data.shape[0], -1)

    return 0.5*torch.mean((mdata-mmu)**2/(torch.exp(msigma)+EPSILON) + msigma)


def kernel(a, b): #N x M, K x M
    dist1 = (a**2).sum(dim=1).unsqueeze(1).expand(-1, b.shape[0]) #N x C
    dist2 = (b**2).sum(dim=1).unsqueeze(0).expand(a.shape[0], -1) #N x C
    dist3 = torch.mm(a, b.transpose(0, 1))
    dist = dist1 + dist2 - 2 * dist3
    return torch.mean(torch.exp(-dist))
