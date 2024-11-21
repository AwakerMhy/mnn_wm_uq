import torch
import numpy as np
import matplotlib.pyplot as plt 


def plot_ellipse(mean, cov, n_std=1, rescale=1, linewidth=5, s=20):
    #eig_value , eig_vector = torch.symeig(cov, eigenvectors = True)

    if isinstance(cov, np.ndarray):
        cov = torch.from_numpy(cov)
    
    if isinstance(mean, np.ndarray):
        mean = torch.from_numpy(mean)

    eig_value , eig_vector = torch.linalg.eigh(cov)
    eig_value = eig_value.numpy()        
    eig_vector = eig_vector.numpy()

    th = np.linspace(0,2*np.pi,101)
    x = np.sqrt(eig_value[0])*np.cos(th)*n_std/rescale #+ u_mat[index,0,x1]
    y = np.sqrt(eig_value[1])*np.sin(th)*n_std/rescale# + u_mat[index,0,x2]

    r = np.vstack((x,y))
    r = np.matmul(eig_vector,r)
    x = r[0,:]+ mean[0].item()
    y = r[1,:]+ mean[1].item()
    plt.plot(x,y,color='orangered',linewidth=linewidth,alpha=0.9)
    plt.scatter(mean[0].item(), mean[1].item(), s=s, c='black')
