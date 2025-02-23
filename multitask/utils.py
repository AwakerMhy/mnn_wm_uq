import numpy as np
import torch
from .task import generate_trials

def gen_feed_dict(trial, hp):
    """Generate feed_dict for session run."""
    if hp['in_type'] == 'normal':
        feed_dict = {'x': trial.x,
                     'y': trial.y,
                     'c_mask': trial.c_mask}
    elif hp['in_type'] == 'multi':
        n_time, batch_size = trial.x.shape[:2]
        new_shape = [n_time,
                     batch_size,
                     hp['rule_start']*hp['n_rule']]

        x = np.zeros(new_shape, dtype=np.float32)
        for i in range(batch_size):
            ind_rule = np.argmax(trial.x[0, i, hp['rule_start']:])
            i_start = ind_rule*hp['rule_start']
            x[:, i, i_start:i_start+hp['rule_start']] = \
                trial.x[:, i, :hp['rule_start']]

        feed_dict = {'x': x,
                     'y': trial.y,
                     'c_mask': trial.c_mask}
    else:
        raise ValueError()
    return feed_dict

def popvec(y):
    pref = np.arange(0, 2*np.pi, 2*np.pi/y.shape[-1])  
    temp_sum = y.sum(axis=-1)
    temp_cos = np.sum(y*np.cos(pref), axis=-1)/temp_sum
    temp_sin = np.sum(y*np.sin(pref), axis=-1)/temp_sum
    loc = np.arctan2(temp_sin, temp_cos)
    return np.mod(loc, 2*np.pi)


def torch_popvec(y):
    if y.min() <= 0. : 
        y = y - y.min() + 0.1
    device = y.device
    num_units = y.shape[-1]
    pref = torch.arange(0, 2 * np.pi, 2 * np.pi / num_units,device=device)
    cos_pref = torch.cos(pref)
    sin_pref = torch.sin(pref)
    temp_sum = y.sum(dim=-1)
    temp_cos = (y * cos_pref).sum(dim=-1) / temp_sum
    temp_sin = (y * sin_pref).sum(dim=-1) / temp_sum 
    loc = torch.atan2(temp_sin, temp_cos)
    return torch.remainder(loc, 2*np.pi)


def esti_output_stats(model, mu, cov, y_loc, num_samples=200):
    device = mu.device
    U_s,S_s,_ = torch.linalg.svd(cov[0,...])
    L_s = torch.matmul(U_s, torch.diag_embed(torch.sqrt(S_s)))
    normal_samples = torch.randn(mu.shape[1],num_samples,mu.shape[2],device=device)
    samples = torch.einsum('bij,bki->bkj',L_s,normal_samples)+mu[-1].unsqueeze(1)
    y_s = model.g_func(samples)
    cov_s = (((y_s.unsqueeze(3)*y_s.unsqueeze(2)).mean(1))-\
     (y_s.unsqueeze(3).mean(1)*y_s.unsqueeze(2).mean(1)))
    uncern_s = torch.log(torch.linalg.svd(cov_s)[1]).sum(1).cpu().numpy()
    return uncern_s


def get_perf(y_hat, y_loc):
    if len(y_hat.shape) != 3:
        raise ValueError('y_hat must have shape (Time, Batch, Unit)')
    y_loc = y_loc[-1]
    y_hat = y_hat[-1]
    y_hat_fix = y_hat[..., 0]
    if y_hat.shape[-1]==33:
        y_hat_loc = popvec(y_hat[..., 1:])
    else:
        y_hat_loc = popvec(y_hat)
    fixating = y_hat_fix > 0.5
    original_dist = y_loc - y_hat_loc
    dist = np.minimum(abs(original_dist), 2*np.pi-abs(original_dist))
    corr_loc = dist < 0.2*np.pi
    should_fix = y_loc < 0
    perf = should_fix * fixating + (1-should_fix) * corr_loc * (1-fixating)
    return perf

def get_perf_torch(y_hat, y_loc):
    if len(y_hat.shape) != 3:
        raise ValueError('y_hat must have shape (Time, Batch, Unit)')
    y_loc = y_loc[-1]
    y_hat = y_hat[-1]
    y_hat_fix = y_hat[..., 0]
    y_hat_loc = torch_popvec(y_hat[..., 1:])
    fixating = (y_hat_fix > 0.5).float()
    original_dist = y_loc - y_hat_loc
    dist = torch.minimum(abs(original_dist), 2*np.pi-abs(original_dist))
    corr_loc = (dist < 0.2*np.pi).float()
    should_fix = (y_loc < 0).float()
    perf = should_fix * fixating + (1-should_fix) * corr_loc * (1-fixating)
    return perf, dist

def get_perf_uq_torch(y_hat, y_loc):
    if len(y_hat.shape) != 3:
        raise ValueError('y_hat must have shape (Time, Batch, Unit)')
    y_loc = y_loc[-1]
    y_hat = y_hat[-1]
    y_hat_loc = torch_popvec(y_hat)
    original_dist = y_loc - y_hat_loc
    dist = torch.minimum(abs(original_dist), 2*np.pi-abs(original_dist))
    should_not_fix = (y_loc >= 0)
    return dist, should_not_fix


def evaluation(model, hp, N, uq=False, extra_noise_sigma=0.,corr=False):
    device = model.device
    perf_test_s = []
    dist_s_s = []
    feed_dict_s = []
    if uq:
        uq_corr_s = []
        error_s_s = []
        uncern_s_s = []
        output_mu_s=[]
        output_cov_s=[]
    with torch.no_grad():
        for rule_test in hp['rules']:
            trial = generate_trials(
                            rule_test, hp, 'random', batch_size=hp['batch_size_test'])
            feed_dict = gen_feed_dict(trial, hp)
            feed_dict_s.append(feed_dict)
            batch_x, _, _ =\
                torch.from_numpy(feed_dict['x']).to(device), torch.from_numpy(feed_dict['y']).to(device), torch.from_numpy(feed_dict['c_mask']).to(device)

            mu_s, y_hat_test, output_cov = model(batch_x,mu=torch.zeros(1,1,N,device=device),
                        cov=torch.eye(N,device=device).reshape(1,1,N,N)*hp['sigma_init']**2,extra_noise_sigma=extra_noise_sigma,
                        out_mu=True,corr=corr)
            if uq:
                error_s, should_not_fixed = get_perf_uq_torch(y_hat_test, torch.from_numpy(trial.y_loc).to(device))
                error_s = error_s.cpu().numpy()
                should_not_fixed = should_not_fixed.cpu().numpy()
                uncern_s = esti_output_stats(model,mu_s[:,:,1:],output_cov[:,:,1:,1:],torch.from_numpy(trial.y_loc).to(device))           
                output_mu_s.append(y_hat_test[-1].detach().cpu().numpy())
                output_cov_s.append(output_cov.detach().cpu().numpy())
                uq_corr = np.corrcoef(error_s[should_not_fixed],uncern_s[should_not_fixed])[0,1]
                uq_corr_s.append(uq_corr)
                error_s_s.append(error_s)
                uncern_s_s.append(uncern_s)
            perf_test, dist_s = get_perf_torch(y_hat_test, torch.from_numpy(trial.y_loc).to(device))

            perf_test = perf_test.mean()
            dist_s = dist_s.mean()
            perf_test_s.append(perf_test.item())
            dist_s_s.append(dist_s.item())
            
    if uq:
        output_mu_s=np.stack(output_mu_s)
        output_cov_s=np.vstack(output_cov_s)
        return perf_test_s, hp['rules'], np.array(uq_corr_s), np.array(error_s_s), np.array(uncern_s_s),\
         output_cov_s, feed_dict_s, output_mu_s
    else:
        return perf_test_s, dist_s_s, hp['rules'], feed_dict_s

