import torch
import numpy as np
from .activation import MomentActivation

def manifold_func(gamma, a=2, h=0, A=1.2):
    factor = A/(1+h/(a-1))
    return factor*(np.cos(gamma)+h/(a-1)*np.cos((a-1)*gamma)),\
           factor*(np.sin(gamma)-h/(a-1)*np.sin((a-1)*gamma))


class MnnRnn:
    def __init__(self, N=1000, g=1, tau=1, dt=0.1, device='cpu', sparse=False):
        self.device = device

        self.N = N
        self.tau = tau
        self.g = g 
        self.dt = dt
        self.phi = MomentActivation()
        self.sparse = sparse

        theta_s = torch.linspace(0, 2*np.pi, N)
        self.W_fb = torch.vstack([torch.cos(theta_s),torch.sin(theta_s)]).T.to(self.device)

        self.W_out = torch.randn([N,2],device=self.device)/np.sqrt(N)
        self.J =  torch.randn([N,N],device=self.device)/np.sqrt(N)

        if sparse > 0:
            mask = (torch.rand([N,N],device=self.device)>sparse).float()
            self.J *= mask

        self.mu = torch.rand([N, 1],device=self.device)
        self.cov = torch.zeros([N, N, 1],device=self.device)
        self.cov[np.arange(N),np.arange(N),0] = 0.001
    
    def reset(self, batch_size=1):
        self.mu = torch.rand([self.N, batch_size],device=self.device)
        self.cov = torch.zeros([self.N, self.N, batch_size],device=self.device)
        self.cov[np.arange(self.N),np.arange(self.N),:] = 0.001
        self.W = self.g * self.J + self.W_fb @ self.W_out.T

    @classmethod
    def cov_mat(self, weight, cov):
        return torch.einsum('lkj,ij->ikl',torch.einsum('ij,jkl->lik',weight,cov),weight)

    def update(self, mu_s=0, sigma_s=0):
        # u.shape: N x batch size
        ubar = self.W @ self.mu + mu_s

        Cbar = self.cov_mat(self.W, self.cov)
        Cbar[np.arange(self.N),np.arange(self.N),:] += sigma_s ** 2

        mu, cov = self.phi(ubar.T,Cbar.permute(2,0,1))
        mu, cov = mu.T, cov.permute(1,2,0)

        self.mu = self.mu + self.dt / self.tau * (-self.mu + mu)
        self.cov = self.cov + self.dt / self.tau * (-self.cov + cov)

    def update_z(self, z, mu_s=0, sigma_s=0):
        # z.shape: 2 x batch size
        ubar = self.g * self.J @ self.mu + self.W_fb @ z + mu_s
        Cbar = self.cov_mat(self.g*self.J, self.cov)

        Cbar[np.arange(self.N),np.arange(self.N),:] += sigma_s ** 2

        mu, cov = self.phi(ubar.T,Cbar.permute(2,0,1))
        mu, cov = mu.T, cov.permute(1,2,0)

        self.mu = self.mu + self.dt / self.tau * (-self.mu + mu)
        self.cov = self.cov + self.dt / self.tau * (-self.cov + cov)

    # the norm of the velocity
    def Q_func(self, mu, cov, mu_s=0, sigma_s=0):
        ubar = self.W @ mu + mu_s
        Cbar = self.cov_mat(self.W, cov)

        Cbar[np.arange(self.N),np.arange(self.N),:] += sigma_s ** 2
        mu_, cov_ = self.phi(ubar.T,Cbar.permute(2,0,1))
        mu_, cov_ = mu_.T, cov_.permute(1,2,0)
        v_mu = -mu + mu_
        v_cov = -cov + cov_
        return (v_mu**2).sum() + (v_cov**2).sum()
    def decode(self, mu, cov=None):
        if cov is None:
            return self.W_out.T @ mu
        else:
            return self.W_out.T @ mu, self.cov_mat(self.W_out.T, self.cov)
    
# MNN that only considers variance propagation
class VnnRnn:
    def __init__(self, N=1000, g=1, tau=1, dt=0.1, device='cpu', sparse=False):
        self.device = device

        self.N = N
        self.tau = tau
        self.g = g 
        self.dt = dt
        self.phi = MomentActivation()
        self.sparse = sparse

        theta_s = torch.linspace(0, 2*np.pi, N)
        self.W_fb = torch.vstack([torch.cos(theta_s),torch.sin(theta_s)]).T.to(self.device)

        self.W_out = torch.randn([N,2],device=self.device)/np.sqrt(N)
        self.J =  torch.randn([N,N],device=self.device)/np.sqrt(N)

        if sparse > 0:
            mask = (torch.rand([N,N],device=self.device)>sparse).float()
            self.J *= mask

        self.mu = torch.rand([N, 1],device=self.device)
        self.cov = torch.zeros([N, N, 1],device=self.device)
        self.cov[np.arange(N),np.arange(N),0] = 0.001
    
    def reset(self, batch_size=1):
        self.mu = torch.rand([self.N, batch_size],device=self.device)
        self.cov = torch.zeros([self.N, self.N, batch_size],device=self.device)
        self.cov[np.arange(self.N),np.arange(self.N),:] = 0.001
        self.W = self.g * self.J + self.W_fb @ self.W_out.T

    #@classmethod
    def cov_mat(self, weight, cov):
        # l: batch size
        result_dim = weight.shape[0]
        diag = torch.matmul(weight*weight, cov[np.arange(self.N),np.arange(self.N),:])
        result = torch.zeros([result_dim,result_dim,diag.shape[-1]],device=self.device)
        result[np.arange(result_dim),np.arange(result_dim),:] = diag
        return result


    def update(self, mu_s=0, sigma_s=0):
        # u.shape: N x batch size
        ubar = self.W @ self.mu + mu_s
        Cbar = self.cov_mat(self.W, self.cov)
        Cbar[np.arange(self.N),np.arange(self.N),:] += sigma_s ** 2

        mu, cov = self.phi(ubar.T,Cbar.permute(2,0,1))
        mu, cov = mu.T, cov.permute(1,2,0)

        self.mu = self.mu + self.dt / self.tau * (-self.mu + mu)
        self.cov = self.cov + self.dt / self.tau * (-self.cov + cov)

    def update_z(self, z, mu_s=0, sigma_s=0):
        # z.shape: 2 x batch size
        ubar = self.g * self.J @ self.mu + self.W_fb @ z + mu_s
        Cbar = self.cov_mat(self.g*self.J, self.cov)

        Cbar[np.arange(self.N),np.arange(self.N),:] += sigma_s ** 2

        mu, cov = self.phi(ubar.T,Cbar.permute(2,0,1))
        mu, cov = mu.T, cov.permute(1,2,0)

        self.mu = self.mu + self.dt / self.tau * (-self.mu + mu)
        self.cov = self.cov + self.dt / self.tau * (-self.cov + cov)

    # the norm of the velocity
    def Q_func(self, mu, cov, mu_s=0, sigma_s=0):
        ubar = self.W @ mu + mu_s
        Cbar = self.cov_mat(self.W, cov)

        Cbar[np.arange(self.N),np.arange(self.N),:] += sigma_s ** 2
        mu_, cov_ = self.phi(ubar.T,Cbar.permute(2,0,1))
        mu_, cov_ = mu_.T, cov_.permute(1,2,0)
        v_mu = -mu + mu_
        v_cov = -cov + cov_
        return (v_mu**2).sum() + (v_cov**2).sum()


    def decode(self, mu, cov=None):
        if cov is None:
            return self.W_out.T @ mu
        else:
            return self.W_out.T @ mu, self.cov_mat(self.W_out.T, self.cov)
    