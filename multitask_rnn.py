import numpy as np
from utils.log import Logger
from datetime import datetime

import torch
import torch.optim as optim
import os
from multitask.models import MnnRnn
from multitask.utils import gen_feed_dict,evaluation
from multitask.task import get_num_ring, get_num_rule, rules_dict, generate_trials

import pickle

config = {
        'N':128, 
        'max_steps':int(1e6), 
        'display_step':500,
        'ruleset':'without_match',
        'rule_trains':None,
        'rule_prob_map':None,
        'seed':0,
        'device':'cuda:0',
        'extra_noise_sigma':0.0,
        'use_grad_clip':True,
        't_ref':0.0,
        'init_weight': 10, 
        'init_weight2': 8,  
        'init_bias': 0.7, 
        'sigma_input':True, 
        'scale': 1., 
        'bias_mu': 0, 
        'base_sigma': 0.,
        'simple': ['fdgo', 'reactgo', 'delaygo', 'fdanti', 'reactanti', 'delayanti'],
        'skip_rate':0.8
        }

t_ref = config['t_ref']
init_weight = config['init_weight']
init_weight2 = config['init_weight2']
init_bias = config['init_bias']
N=config['N'] 
max_steps=config['max_steps'] 
display_step=config['display_step'] 
ruleset=config['ruleset']
rule_trains=config['rule_trains'] 
rule_prob_map=config['rule_prob_map'] 
seed=config['seed']
device = config['device']
skip_rate = config['skip_rate']


num_ring = get_num_ring(ruleset)
n_rule = get_num_rule(ruleset)
    
n_eachring = 32  
n_input, n_output = 1+num_ring*n_eachring+n_rule, n_eachring+1 
hp = {  'sigma_init': 0.1, 
        'batch_size_train': 64,
        'batch_size_test': 512,
        'in_type': 'normal',
        'rnn_type': 'LeakyRNN',
        'use_separate_input': False,
        'loss_type': 'lsq',
        'optimizer': 'adam',
        'tau': 100,
        'dt': 20,
        'alpha': 0.1,
        'sigma_rec': 0.01,
        'sigma_x': 0.01, 
        'l2_weight': 0,
        'target_perf': 1.,
        'n_eachring': n_eachring,
        'num_ring': num_ring,
        'n_rule': n_rule,
        'rule_start': 1+num_ring*n_eachring,
        'n_input': n_input,
        'n_output': n_output,
        'n_rnn': 256,
        'ruleset': ruleset,
        'save_name': 'test',
        'learning_rate':0.0005,
        'c_intsyn': 0,
        'ksi_intsyn': 0,
        }

sigma_input = hp['sigma_x'] if config['sigma_input'] else None
hp['seed'] = seed
hp['rng'] = np.random.RandomState(seed)
if rule_trains is None:
    hp['rule_trains'] = rules_dict[ruleset]
else:
    hp['rule_trains'] = rule_trains
hp['rules'] = hp['rule_trains']
if rule_prob_map is None:
    rule_prob_map = dict()
hp['rule_probs'] = None
if hasattr(hp['rule_trains'], '__iter__'):
    rule_prob = np.array(
            [rule_prob_map.get(r, 1.) for r in hp['rule_trains']])
    hp['rule_probs'] = list(rule_prob/np.sum(rule_prob))


timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
save_name = f'{timestamp}'

workdir = f'multitask_result/{save_name}'

os.makedirs(workdir)
logger = Logger(f'{workdir}/log.log')
logger.log(config)
logger.log(hp)


with open(f'{workdir}/settings.pickle','wb+') as f:
    pickle.dump([config,hp],f)


model = MnnRnn(input_dim=n_input, output_dim=n_output, N=N,
                tau=hp['tau'], dt=hp['dt'], sigma_rec=hp['sigma_rec'], device=device,
                t_ref=t_ref, init_weight=init_weight, init_weight2=init_weight2, init_bias=init_bias, sigma_input=sigma_input,
                        scale=config['scale'],bias_mu=config['bias_mu'],base_sigma=config['base_sigma'])
model.train()
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=hp['learning_rate'],weight_decay=hp['l2_weight'])


loss_s = []
acc_s_s = []
dist_s_s = []
for step in range(max_steps):
    rule_train_now = hp['rng'].choice(hp['rule_trains'],p=hp['rule_probs'])
    trial = generate_trials(
        rule_train_now, hp, 'random',
        batch_size=hp['batch_size_train'])

    feed_dict = gen_feed_dict(trial, hp)

    batch_x, batch_y, c_mask =\
        torch.from_numpy(feed_dict['x']).to(device), torch.from_numpy(feed_dict['y']).to(device), \
            torch.from_numpy(feed_dict['c_mask']).to(device)
    optimizer.zero_grad()
    outputs_seq, output_cov = model(batch_x,mu=torch.zeros(1,1,N,device=device), 
                                    cov=torch.eye(N,device=device).reshape(1,1,N,N)*hp['sigma_init']**2,
                                    skip_rate=skip_rate)


    loss = ((outputs_seq-batch_y).reshape(-1,batch_y.shape[-1])*c_mask).pow(2).mean()
    loss.backward()
    if config['use_grad_clip']:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step() 
    if step % 100 == 0:
        logger.log('step: {}, loss {:.4f}'.format(step, loss.item()))
        if step % display_step == 0:

            acc_s, dist_s, rule_s,feed_dict_s = evaluation(model, hp, N)
            logger.log('------------------------------acc------------------------------')
            logger.log(', '.join(['{}: {:.2f}'.format(rule, acc) for rule, acc in zip(rule_s, acc_s)]))
            logger.log('------------------------------distance-------------------------')
            logger.log(', '.join(['{}: {:.2f}'.format(rule, acc) for rule, acc in zip(rule_s, dist_s)]))
            logger.log('------------------------------corr-----------------------------')

            acc_s, rule_s, uq_corr_s, error_s_s, uncern_s_s, output_cov_s, feed_dict_s, output_mu_s = evaluation(model, hp, N, uq=True)
            total_corr = np.corrcoef(error_s_s.reshape(-1),uncern_s_s.reshape(-1))[0,1]
            excuded_list=[]
            for i,rule in enumerate(rule_s):
                if rule not in config['simple']:
                    excuded_list.append(i)
            total_corr_excluded = np.corrcoef(error_s_s[excuded_list,:].reshape(-1),uncern_s_s[excuded_list,:].reshape(-1))[0,1]

            logger.log(', '.join(['{}: {:.2f}'.format(rule, corr) for rule, corr in zip(rule_s, uq_corr_s)]))
            logger.log('{}, total corr: {:.4f}'.format(save_name, total_corr))
            logger.log('{}, total corr excluded simple: {:.4f}'.format(save_name, total_corr_excluded))

            acc_s_s.append(acc_s)
            dist_s_s.append(dist_s)

        loss_s.append(loss.item())
    if step % 10000 == 0:
        torch.save(model.state_dict(), f'{workdir}/model_{step}.pth')    
    torch.save(model.state_dict(), f'{workdir}/model.pth')


np.savez(f'{workdir}/result',loss_s=np.array(loss_s), acc_s_s=np.array(acc_s_s), dist_s_s=np.array(dist_s_s))