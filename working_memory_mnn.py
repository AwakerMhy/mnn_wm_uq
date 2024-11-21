import torch
import numpy as np
from mnn import MnnRnn, VnnRnn, manifold_func
from sklearn.linear_model import Ridge
from utils.log import Logger
from datetime import datetime
import os
import random
import matplotlib

matplotlib.use('Agg')

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
config = {
        'var_only':False, # wheter ignore neural correlations
        'dt':0.1,
        'tau':1.,
        'N': 200, # 100 200 400 500 1000
        'M': 10, # 10 20 30 40
        'h': 0,
        'a': 2,
        'A': 1.2,
        'g': 3., 
        'sparse':0., 
        'alpha':5e-3, # coefficient of L2
        'mu_s':0.0,
        'sigma_s':1., # training time sigma
        'sigma_s_2':1., # inference time sigma
        'ensemble_num':1, 
        'training_steps':1000, # 1000
        'stimuli_steps': 500, 
        'total_steps': 7500, 
        'visual_num':1,
        'sample_num':500, 
        'batch_size':500, 
        'input_noise_level': 1., 
        'device':'cuda:0',
        'seed':[0,1,2,3,4,5,6,7,8,9],
        }


if isinstance(config['seed'], list):
    seeds = config['seed']
else:
    seeds = [config['seed']]

main_workdir = f'results/{timestamp}'


for seed in seeds:
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    workdir = f'{main_workdir}/seed_{seed}'
    os.makedirs(workdir)
    logger = Logger(f'{workdir}/log.log')

    config['seed'] = seed

    logger.log(config)

    if config['var_only']:
        nets = [VnnRnn(config['N'], config['g'], tau=config['tau'], dt=config['dt'], 
                    device=config['device'], sparse=config['sparse']) for i in range(config['ensemble_num'])]
    else:
        nets = [MnnRnn(config['N'], config['g'], tau=config['tau'], dt=config['dt'], 
                    device=config['device'], sparse=config['sparse']) for i in range(config['ensemble_num'])]


    with torch.no_grad():
        for i,net in enumerate(nets):
            gamma_s = np.linspace(0, 2*np.pi, config['M'])
            samples = torch.tensor([manifold_func(gamma, config['a'], config['h'], config['A']) for gamma in gamma_s]).float().to(config['device'])
            net.reset(samples.shape[0])
            for step in range(config['training_steps']):
                net.update_z(samples.T,mu_s=config['mu_s'],sigma_s=config['sigma_s'])
                if step % 100 == 0:
                    print('ensemble:', i, 'step: ',step)
            fixed_point_s = net.mu.T

            fixed_point_s = fixed_point_s.cpu().numpy()
            samples = samples.cpu().numpy()

            solver = Ridge(alpha=config['alpha'],fit_intercept=False)

            solver.fit(fixed_point_s, samples)
            W_out = solver.coef_.T
            net.W_out = torch.from_numpy(W_out).float().to(config['device'])
            net.reset()


    decoded_mu_s = []
    decoded_cov_s = []

    batch_size = config['batch_size']
    total_size = config['sample_num']

    assert total_size % batch_size == 0

    fixed_mu_s = []
    fixed_cov_s = []

    with torch.no_grad():
        all_gamma_s = np.linspace(0, 2*np.pi, config['sample_num'])
        all_samples = torch.tensor([manifold_func(gamma, config['a'], config['h'], config['A']) for gamma in all_gamma_s]).float().to(config['device'])
        sigma_s_base = torch.ones([1,config['sample_num']],device=config['device'])
        noise_std = torch.rand([1,config['sample_num']],device=config['device'])

        input_noise = torch.randn(config['stimuli_steps'],config['N'],total_size,device=config['device'])

        for i, net in enumerate(nets):

            decoded_mu_s_batch = []
            decoded_cov_s_batch = []

            fixed_mu_batch = []
            fixed_cov_batch = []

            for batch_index in range(total_size//batch_size):
                batch_sample = all_samples[batch_index * batch_size:(batch_index+1) * batch_size,:]
                mu_s=net.W_fb @ (batch_sample.T)
                net.reset(batch_size)            

                for step in range(config['total_steps']):
                    if step < config['stimuli_steps']:
                        noise = input_noise[step,:,batch_index * batch_size:(batch_index+1) * batch_size]*\
                                noise_std[:,batch_index * batch_size:(batch_index+1) * batch_size]*config['input_noise_level']   
                        net.update(mu_s=mu_s + noise, sigma_s=config['sigma_s_2']*sigma_s_base[:,batch_index * batch_size:(batch_index+1) * batch_size])
                    else:
                        net.update(mu_s=config['mu_s'], sigma_s=config['sigma_s_2']*sigma_s_base[:,batch_index * batch_size:(batch_index+1) * batch_size])
                    if step % 100 == 0:
                        print('batch index: ', batch_index, 'ensemble: ', i, 'step: ',step)

                fixed_mu_batch.append(net.mu.cpu())
                fixed_cov_batch.append(net.cov.cpu())

                decoded_mu, decoded_cov = net.decode(net.mu, net.cov)
                decoded_mu, decoded_cov = decoded_mu.cpu(), decoded_cov.cpu()

                decoded_mu_s_batch.append(decoded_mu.clone())
                decoded_cov_s_batch.append(decoded_cov.clone())

            fixed_mu_batch = torch.cat(fixed_mu_batch,dim=-1)
            fixed_cov_batch = torch.cat(fixed_cov_batch,dim=-1)

            fixed_mu_s.append(fixed_mu_batch)
            fixed_cov_s.append(fixed_cov_batch)


            decoded_mu_s_batch = torch.cat(decoded_mu_s_batch,dim=-1)
            decoded_cov_s_batch = torch.cat(decoded_cov_s_batch,dim=-1)
            
            decoded_mu_s.append(decoded_mu_s_batch)
            decoded_cov_s.append(decoded_cov_s_batch)


    all_samples = all_samples.cpu()
    decoded_mu_s = torch.stack(decoded_mu_s).cpu()
    decoded_cov_s = torch.stack(decoded_cov_s).cpu()

    fixed_mu_s = torch.stack(fixed_mu_s).cpu()
    fixed_cov_s = torch.stack(fixed_cov_s).cpu()

    noise_std = noise_std.cpu()

    W_out_s = []
    W_fb_s = []
    J_s = []

    for net in nets:
        W_out_s.append(net.W_out.cpu().numpy())
        W_fb_s.append(net.W_fb.cpu().numpy())
        J_s.append(net.J.cpu().numpy())

    W_out_s = np.stack(W_out_s)
    W_fb_s = np.stack(W_fb_s)
    J_s = np.stack(J_s)

    all_samples/=all_samples.pow(2).sum(dim=1,keepdim=True).pow(0.5)
    decoded_mu = decoded_mu_s.mean(0)
    decoded_cov = decoded_cov_s.mean(0)
    decoded_mu/=decoded_mu.pow(2).sum(dim=0,keepdim=True).pow(0.5)

    np.savez(f'{workdir}/results',
            W_fb=W_fb_s, W_out=W_out_s,J=J_s, 
            all_samples=all_samples.numpy(), 
            decoded_mu_s=decoded_mu_s.numpy(), 
            decoded_cov_s=decoded_cov_s.numpy(),
            noise_std=noise_std.numpy(),
            fixed_mu_s=fixed_mu_s.numpy(),
            fixed_cov_s=fixed_cov_s.numpy())