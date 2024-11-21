import torch
import numpy as np
import tqdm

class LIF_ensemble: 
    def __init__(self,num_neurons, num_ensemble, weight, dt=1e-1, tau=10., device='cpu'):
        self.num_neurons = num_neurons
        self.num_ensemble = num_ensemble
        self.device=device
        self.weight = weight
        self.V_reset = torch.tensor(0.).to(self.device)
        self.V_th  = torch.tensor(20.).to(self.device)
        self.T_ref = torch.tensor(5.).to(self.device)
        self.dt = torch.tensor(dt).to(self.device)
        self.L = torch.tensor(0.05).to(self.device)
        self.tau = tau
        self.initialize()
        
    def initialize(self):
        self.V = (torch.rand(self.num_ensemble,self.num_neurons).float().to(self.device)*self.V_th)
        self.spike = (torch.zeros(self.num_ensemble,self.num_neurons).float()).to(self.device)
        self.t_ik_last = (torch.ones(self.num_ensemble,self.num_neurons).float()*-10).to(self.device)
        self.t = (torch.tensor(0.)).to(self.device)
        self.I_syn = (torch.zeros(self.num_ensemble,self.num_neurons).float()).to(self.device)
    def run(self, I_ext=torch.tensor([0])):
        self.I_syn = self.I_syn + (-self.I_syn/self.tau) * self.dt + \
        torch.bmm(self.weight,self.spike.float().unsqueeze(2)).squeeze(2)/self.tau        
        Vi_normal = self.V + self.dt * (- self.L * self.V  + self.I_syn) + I_ext.to(self.device).reshape(1,-1) 
        is_not_saturated = (self.t_ik_last + self.T_ref) <= self.t
        V = torch.where(is_not_saturated, Vi_normal, self.V_reset)
        self.spike = torch.ge(V, self.V_th) 
        self.V = torch.min(V, self.V_th)  
        self.t_ik_last = torch.where(self.spike, self.t, self.t_ik_last)  
        self.t+=self.dt

def readout(mean, cov, true, W_out):
    decoded_mu = (W_out.T @ mean).numpy()
    decode_theta = np.arctan2(decoded_mu[0],decoded_mu[1])
    if decode_theta<0:
        decode_theta+=2*np.pi

    dist_ = np.abs(decode_theta - true)
    angle_error = (np.abs(dist_)+np.abs(dist_-2*np.pi) - np.abs(np.abs(dist_)-np.abs(dist_-2*np.pi)))/2
    decoded_cov = torch.mm(torch.mm(W_out.T, cov),W_out)

    uncern1 = decoded_cov[0,0]*decoded_cov[1,1] - decoded_cov[1,0]*decoded_cov[1,0]
    return angle_error, uncern1

def test(instance_index, start, time_steps, length):
    spikes = []
    cue = torch.from_numpy((W_fb @ result['all_samples'][instance_index])).reshape(1,-1).to(device)
    for i in range(time_steps):
        if i < exrt_time:
            snn.run(I_ext=cue+xi_ext*np.sqrt(dt) * torch.randn([N],device=device))
        else:
            snn.run(I_ext=std_ext*np.sqrt(dt) * torch.randn([N],device=device))
        if i >= start and i < time_steps:
            spikes.append(snn.spike.cpu())

    spikes = torch.stack(spikes)
    spikes = spikes.float()

    mean = spikes.mean(1).mean(0)

    n = 0
    i = 0
    x = 0
    x_2 = 0
    num = 0
    for n in range((time_steps-start)//length-1):
        for i in range(num_ensemble):
            num+=1
            x+=spikes[n*length:(n+1)*length,i,:].mean(0)
            x_2+=spikes[n*length:(n+1)*length,i,:].mean(0).reshape(-1,1)*spikes[n*length:(n+1)*length,i,:].mean(0).reshape(1,-1)
    cov = x_2/num - x.reshape(-1,1) * x.reshape(1,-1) / num**2
    return mean, cov

np.random.seed(0)
torch.manual_seed(0)


result_name = 'example'

result = np.load(f'{result_name}/results.npz')

all_samples = result['all_samples']
all_theta_s = np.arctan2(all_samples[:,0], all_samples[:,1])
all_theta_s[all_theta_s<0]+=2*np.pi

g = 3.
W_fb, W_out, J = result['W_fb'][0,...], result['W_out'][0,...], result['J'][0,...]
W = np.matmul(W_fb ,W_out.T) +g*J 
W_out = torch.from_numpy(W_out)


N = 200
dt = 0.1
num_ensemble = 1
device = 'cuda:0'
time_steps = 7500
exrt_time = 500
std_ext = 1.
xi_ext = 1.
snn = LIF_ensemble(num_neurons=N,dt=dt,num_ensemble=num_ensemble,weight=\
                    torch.from_numpy(W).unsqueeze(0).expand(num_ensemble,N,N).to(device),device=device)

start = 2000
length = 500

angle_error_s = []
uncern1_s = []
mean_s = []
cov_s = []
for instance_index in tqdm.tqdm(range(500)):
    mean, cov = test(instance_index, start, time_steps, length)
    mean_s.append(mean.numpy())
    cov_s.append(cov.numpy())
    true = all_theta_s[instance_index]
    angle_error, uncern1 = readout(mean, cov, true, W_out)
    angle_error_s.append(angle_error)
    uncern1_s.append(uncern1)

mean_s = np.array(mean_s)
cov_s = np.array(cov_s)

corr1 = np.corrcoef(angle_error_s,uncern1_s)[0,1]

print('corr1: {:.4f}.'.format(corr1))
