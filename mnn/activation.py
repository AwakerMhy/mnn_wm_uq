import torch
from torch import Tensor
from typing import Tuple
from . import functional
from  .mnn_core.mnn_pytorch import MnnActivateTrio2, MnnActivationNoRho2


class MomentActivation(torch.nn.Module):
    def __init__(self, smuc=False, t_ref=5.0, scale=1., bias_mu=0., base_sigma=0.,out_base_sigma=0.):
        super(MomentActivation, self).__init__()
        self.smuc = smuc
        self.act = MnnActivateTrio2()
        self.act_no_rho = MnnActivationNoRho2()
        self.t_ref = t_ref

        self.scale = scale
        self.bias_mu = bias_mu
        self.base_var = base_sigma**2
        self.out_base_sigma =out_base_sigma


    def forward(self, *args) -> Tuple[Tensor, Tensor]:
        u, cov = functional.parse_input(args)
        u+=self.bias_mu
        if u.size(-1) != 1:
            if self.smuc:
                cov = cov.detach()
                with torch.no_grad():
                    s, r = functional.compute_correlation(cov)
                
                s = torch.sqrt(s*s+self.base_var)
                
                u, s, r = self.act.apply(u, s.detach(), r.detach(),self.t_ref)
                cov = functional.compute_covariance(s.detach(), r.detach())
            else:
                s, r = functional.compute_correlation(cov)
                s = torch.sqrt(s*s+self.base_var)
                u, s, r = self.act.apply(u, s, r, self.t_ref)
                cov = functional.compute_covariance(s, r)
        else:
            cov+=self.base_var
            if self.smuc:
                cov = torch.sqrt(cov).detach()
                u, cov = self.act_no_rho.apply(u, cov, self.t_ref)
                cov = torch.pow(cov, 2)
            else:
                cov = torch.sqrt(cov)
                u, cov = self.act_no_rho.apply(u, cov, self.t_ref)
                cov = torch.pow(cov, 2)
        return u*self.scale+self.out_base_sigma, cov*self.scale*self.scale
