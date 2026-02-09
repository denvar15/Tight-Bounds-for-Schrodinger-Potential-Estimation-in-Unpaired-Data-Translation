import pdb
import math
from torch import nn
from torch.nn.functional import softmax, log_softmax
import torch
import geotorch
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.categorical import Categorical
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.independent import Independent
from torch.distributions.normal import Normal
from tqdm import tqdm


class LightSB_OU(nn.Module):
    def __init__(self, dim=2, n_potentials=5, epsilon=1, is_diagonal=True,
                 sampling_batch_size=1, S_diagonal_init=0.1, b=0.0, m=0.0):
        super().__init__()
        self.is_diagonal = is_diagonal
        self.dim = dim
        self.n_potentials = n_potentials
        self.register_buffer("epsilon", torch.tensor(epsilon))
        self.register_buffer("b", torch.tensor(b)) 
        self.register_buffer("mu", torch.tensor(m))  
        self.sampling_batch_size = sampling_batch_size

        self.log_alpha_raw = nn.Parameter(self.epsilon*torch.log(torch.ones(n_potentials)/n_potentials))
        self.r = nn.Parameter(torch.randn(n_potentials, dim))

        self.S_log_diagonal_matrix = nn.Parameter(torch.log(S_diagonal_init*torch.ones(n_potentials, self.dim)))
        self.S_rotation_matrix = nn.Parameter(
            torch.randn(n_potentials, self.dim, self.dim)
        )
        geotorch.orthogonal(self, "S_rotation_matrix")

    def init_r_by_samples(self, samples):
        assert samples.shape[0] == self.r.shape[0]
        self.r.data = torch.clone(samples.to(self.r.device))

    def get_S(self):
        if self.is_diagonal:
            S = torch.exp(self.S_log_diagonal_matrix)
        else:
            S = (self.S_rotation_matrix*(torch.exp(self.S_log_diagonal_matrix))[:, None, :])@torch.permute(self.S_rotation_matrix, (0, 2, 1))
        return S

    def get_r(self):
        return self.r

    def ou_sigma(self, b):
        if b < 1e-2:
            taylor = 1 - b + (2/3) * b**2 - (1/3) * b**3 + (2/15) * b**4
            return taylor
        else:
            return (1 - torch.exp(-2*b)) / (2*b)

    def get_log_alpha(self):
        return (1/self.epsilon)*self.log_alpha_raw

    @torch.no_grad()
    def forward(self, x):
        S = self.get_S()
        r = self.get_r()
        b = self.b
        mu = self.mu
        epsilon = self.epsilon

        sigma_sq = self.ou_sigma(b)

        log_alpha = self.get_log_alpha()

        decay_factor = torch.exp(-b)

        samples = []
        batch_size = x.shape[0]
        sampling_batch_size = self.sampling_batch_size

        num_sampling_iterations = (
            batch_size // sampling_batch_size if batch_size % sampling_batch_size == 0
            else (batch_size // sampling_batch_size) + 1
        )

        for i in range(num_sampling_iterations):
            sub_batch_x = x[sampling_batch_size*i:sampling_batch_size*(i+1)]

            m_x = (1 - decay_factor) * mu + decay_factor * sub_batch_x

            if self.is_diagonal:
                x_S_x = (m_x[:, None, :] * S[None, :, :] * m_x[:, None, :]).sum(dim=-1)
                x_r = (m_x[:, None, :] * r[None, :, :]).sum(dim=-1)
                r_x = r[None, :, :] + S[None, :] * m_x[:, None, :] 
            else:
                x_S_x = (m_x[:, None, None, :] @ (S[None, :, :, :] @ m_x[:, None, :, None]))[:, :, 0, 0]
                x_r = (m_x[:, None, :] * r[None, :, :]).sum(dim=-1)
                r_x = r[None, :, :] + (S[None, :, :, :] @ m_x[:, None, :, None])[:, :, :, 0] 

            exp_argument = (x_S_x + 2 * x_r) / (2 * epsilon * sigma_sq) + log_alpha[None, :]

            if self.is_diagonal:
                mix = Categorical(logits=exp_argument)
                comp = Independent(Normal(loc=r_x, scale=torch.sqrt(epsilon * sigma_sq * S)[None, :, :]), 1)
                gmm = MixtureSameFamily(mix, comp)
            else:
                mix = Categorical(logits=exp_argument)
                comp = MultivariateNormal(loc=r_x, covariance_matrix=epsilon * sigma_sq * S)
                gmm = MixtureSameFamily(mix, comp)

            sample = gmm.sample()

            samples.append(sample)

        samples = torch.cat(samples, dim=0)
        return samples

    def get_log_potential(self, x):
        S = self.get_S()
        r = self.get_r()
        log_alpha = self.get_log_alpha()
        D = self.dim

        epsilon = self.epsilon

        if self.is_diagonal:
            mix = Categorical(logits=log_alpha)
            comp = Independent(Normal(loc=r, scale=torch.sqrt(self.epsilon * S)), 1)
            gmm = MixtureSameFamily(mix, comp)

            potential = gmm.log_prob(x) + torch.logsumexp(log_alpha, dim=-1)
        else:
            mix = Categorical(logits=log_alpha)
            comp = MultivariateNormal(loc=r, covariance_matrix=self.epsilon * S)
            gmm = MixtureSameFamily(mix, comp)

            potential = gmm.log_prob(x) + torch.logsumexp(log_alpha, dim=-1)

        return potential

    def get_log_C(self, x):
        epsilon = self.epsilon
        b = self.b
        mu = self.mu
        log_alpha = self.get_log_alpha()  
        r = self.get_r() 
        S = self.get_S()

        decay_factor = torch.exp(-b)
        m_x = (1 - decay_factor) * mu + decay_factor * x  

        sigma_sq = self.ou_sigma(b)

        if self.is_diagonal:
            r_term = torch.einsum('kd,bd->bk', r, m_x) / (epsilon * sigma_sq)  

            S_term = 0.5 * torch.einsum('kd,bd->bk', S, m_x**2) / (epsilon * (sigma_sq) )  
        else:
            r_term = torch.einsum('kd,bd->bk', r, m_x) / (epsilon * sigma_sq) 

            S_term = 0.5 * torch.einsum('bd,kde,be->bk', m_x, S, m_x) / (epsilon * (sigma_sq) )

        exp_argument = r_term + S_term + log_alpha[None, :]  
        log_C = torch.logsumexp(exp_argument, dim=-1)

        return log_C

    def sample_at_time_moment(self, x, t):
        t = t.to(x.device)
        y = self(x)
        
        return t*y + (1-t)*x + torch.sqrt(t*(1-t)*self.epsilon)*torch.randn_like(x)

    def ou_sample_at_time_moment(self, x, t):
        t = t.to(x.device)
        y = self(x)  
        
        b = self.b    
        m = self.mu      
        epsilon = self.epsilon  
        
        exp_bt = torch.exp(-b * t)
        exp_b1mt = torch.exp(-b * (1 - t))
        exp_b1 = torch.exp(-b)
        
        mean = (
            m * (1 - exp_bt * exp_b1mt / exp_b1) 
            + x * exp_b1mt * (1 - exp_bt**2) / (1 - exp_b1**2)
            + y * exp_bt * (1 - exp_b1mt**2) / (1 - exp_b1**2)
        )
        
        variance = (1 - torch.exp(-2 * b * t)) * (1 - torch.exp(-2 * b * (1 - t)))
        variance = variance / (1 - torch.exp(-2 * b))
        variance = variance * (epsilon / (2 * b))
        
        return mean + torch.sqrt(variance) * torch.randn_like(x)

    def set_epsilon(self, new_epsilon):
        self.epsilon = torch.tensor(new_epsilon, device=self.epsilon.device)