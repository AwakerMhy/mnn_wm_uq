import numpy as np
import torch
import torch.nn as nn
from mnn.activation import MomentActivation
from mnn import LinearDuo

class MnnRnn(nn.Module):
    def __init__(self, input_dim=85, output_dim=33, N=256, tau=100, 
                 dt=20, sigma_rec=0.05, device='cpu',t_ref=5.0,
                 init_weight=0.5, init_weight2=0.4, init_bias=0.,sigma_input=None,
                 scale=1.,bias_mu=0.,base_sigma=0.):
        super(MnnRnn, self).__init__()
        self.device = device
        self.N = N
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.tau = tau
        self.dt = dt
        self.sigma_rec = sigma_rec
        if sigma_input is not None:
            self.sigma_input = sigma_input
        else:
            self.sigma_input = 0

        self.alpha = dt/tau

        self.phi = MomentActivation(t_ref=t_ref,scale=scale,bias_mu=bias_mu,base_sigma=base_sigma)

        self.linear_rec = LinearDuo(N, N, bias=True)
        self.linear_rec.weight.data = torch.eye(N)*init_weight 
        self.linear_rec.bias.data = torch.zeros(N)+init_bias

        self.linear_in = nn.Linear(input_dim, N, bias=False)
        self.linear_in.weight.data = torch.randn(N, input_dim)*np.sqrt(1/input_dim)

        self.linear_out = nn.Linear(N, output_dim, bias=False)
        self.linear_out.weight.data = torch.randn(output_dim, N)*np.sqrt(1/N)*init_weight2

        self.g_func = torch.sigmoid

    def update(self, inputs, mu, cov, save_Cbar=False, skip_C=False, extra_cov=None):
        var_s = (2/self.alpha)*self.sigma_rec**2
        if skip_C:
            mubar = torch.matmul(self.linear_rec.weight, mu.unsqueeze(-1)).squeeze(-1)
            if self.linear_rec.bias is not None:
                mubar = mubar + self.linear_rec.bias
            mubar = mubar+self.linear_in(inputs)
            Cbar=self.Cbar
        else:
            mubar, Cbar = self.linear_rec(mu, cov)
            mubar = mubar+self.linear_in(inputs)
            Cbar[:,:,np.arange(self.N),np.arange(self.N)]+=var_s
            if self.sigma_input>0:
                cov_input = (self.linear_in.weight @ self.linear_in.weight.T)*self.sigma_input**2
                Cbar = Cbar+cov_input.unsqueeze(0).unsqueeze(0)
        if save_Cbar:
            self.Cbar=Cbar
        if extra_cov is not None:
            input_cov_ = torch.einsum('ij,lmjk->lmik',self.linear_in.weight, extra_cov)
            Cbar+=torch.einsum('ij,lmki->lmkj', self.linear_in.weight.T, input_cov_, )


        mu_new, cov_new = self.phi(mubar[0,...], Cbar[0,...])
        mu = (1-self.alpha) * mu + self.alpha * mu_new.unsqueeze(0)
        cov = (1-self.alpha) * cov_new + self.alpha * cov_new.unsqueeze(0)
        return mu, cov
    
    def forward(self, inputs_seq, mu, cov=None, extra_noise_sigma=0., skip_rate=0, out_mu=False, corr=False):
        # inputs_seq.shape: step x batch_size x input_dim
        if skip_rate>0:
            save_Cbar=True
        else:
            save_Cbar=False
        skip_C=False
        steps = inputs_seq.shape[0]
        outputs_seq = []
        output_mu_s = []
        cov = cov.expand(1,inputs_seq.shape[1],cov.shape[-1],cov.shape[-1])
        for step in range(steps):
            if step>1 and np.random.random()<skip_rate:
                skip_C=True
            else:
                skip_C=False
            mu, cov = self.update(inputs_seq[step]+
                                  torch.randn_like(inputs_seq[step])*extra_noise_sigma, 
                                  mu, cov, save_Cbar=save_Cbar, skip_C=skip_C)
            outputs = self.g_func(self.linear_out(mu))
            output_mu_s.append(self.linear_out(mu))
            outputs_seq.append(outputs)
        if corr:
            var=cov[:,:,np.arange(cov.shape[2]),np.arange(cov.shape[2])]
            var[var<1e-8]=1
            cov/=torch.sqrt(var.unsqueeze(2)*var.unsqueeze(3))
        output_cov = self.linear_out(self.linear_out(cov).permute(0,1,3,2)).permute(0,1,2,3)
        if out_mu:
            return torch.vstack(output_mu_s),torch.vstack(outputs_seq), output_cov
        else:
            return torch.vstack(outputs_seq), output_cov