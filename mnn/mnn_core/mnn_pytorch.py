# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 14:49:43 2020

Copyright 2020 Zhichao Zhu, ISTBI, Fudan University China
"""
from typing import Any
import torch

from .mnn_utils import *

mnn_core_func = Mnn_Core_Func()

def _batch_detach(*args):
    temp = list()
    for i in args:
        temp.append(i.cpu().detach().numpy().astype('float64'))
    return temp


def _batch_flatten(*args):
    temp = list()
    for i in args:
        temp.append(i.flatten())
    return temp


def _batch_reshape(shape, *args):
    temp = list()
    for i in args:
        temp.append(i.reshape(shape).contiguous())
    return temp


def _to_tensor(loc, dtype, *args):
    temp = list()
    for i in args:
        temp.append(torch.from_numpy(i).to(device=loc, dtype=dtype, non_blocking=True))
    return temp

## original
# class MnnActivateTrio(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
#         mean, std, rho = args
#         loc, dtype = mean.device, mean.dtype
#         clone_mean, clone_std = _batch_detach(mean, std)
#         mean_out, std_out, chi = mnn_core_func.fast_forward(clone_mean, clone_std)
#         mean_out, std_out, chi = _to_tensor(loc, dtype, mean_out, std_out, chi)
#         ctx.save_for_backward(mean, std, rho, mean_out, std_out, chi)
#         rho_out = torch.mul(rho, torch.matmul(chi.unsqueeze(-1), chi.unsqueeze(-2)))
#         torch.diagonal(rho_out, dim1=-1, dim2=-2).data.fill_(1.)
#         return mean_out, std_out, rho_out

class MnnActivateTrio(torch.autograd.Function):
    def __init__(self):
        super().__init__()
        # self.mnn_core_func = Mnn_Core_Func()
        # self.mnn_core_func.t_ref = t_ref

    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        mean, std, rho = args
        loc, dtype = mean.device, mean.dtype
        clone_mean, clone_std = _batch_detach(mean, std)
        mean_out, std_out, chi = mnn_core_func.fast_forward(clone_mean, clone_std)
        mean_out, std_out, chi = _to_tensor(loc, dtype, mean_out, std_out, chi)
        ctx.save_for_backward(mean, std, rho, mean_out, std_out, chi)
        rho_out = torch.mul(rho, torch.matmul(chi.unsqueeze(-1), chi.unsqueeze(-2)))
        torch.diagonal(rho_out, dim1=-1, dim2=-2).data.fill_(1.)
        return mean_out, std_out, rho_out

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        mean_grad, std_grad, rho_grad = grad_outputs
        loc, dtype = mean_grad.device, mean_grad.dtype
        mean, std, rho, mean_out, std_out, chi = ctx.saved_tensors
        mean, std, mean_out, std_out, chi = _batch_detach(mean, std, mean_out, std_out, chi)
        mean_grad_mean, mean_grad_std, std_grad_mean, std_grad_std, chi_grad_mean, chi_grad_std = mnn_core_func.fast_backward(mean, std, mean_out, std_out, chi)

        mean_grad_mean, mean_grad_std, std_grad_mean, std_grad_std, chi_grad_mean, chi_grad_std, chi = \
            _to_tensor(loc, dtype, mean_grad_mean, mean_grad_std, std_grad_mean, std_grad_std, chi_grad_mean, chi_grad_std, chi)

        chi_grad_mean = torch.clamp(chi_grad_mean, -1, 1)
        chi_grad_std = torch.clamp(chi_grad_std, -1, 1)
        torch.diagonal(rho, dim1=-1, dim2=-2).data.fill_(0.)
        temp_rho_grad = torch.mul(rho_grad, rho)
        temp_rho_grad = torch.matmul(chi.unsqueeze(-2), temp_rho_grad).squeeze(-2) * 2
        rho_grad_mean = temp_rho_grad * chi_grad_mean
        rho_grad_std = temp_rho_grad * chi_grad_std

        grad_mean = mean_grad * mean_grad_mean + std_grad * std_grad_mean + rho_grad_mean
        grad_std = mean_grad * mean_grad_std + std_grad * std_grad_std + rho_grad_std
        grad_rho = torch.mul(torch.matmul(chi.unsqueeze(-1), chi.unsqueeze(-2)), rho_grad)
        torch.diagonal(grad_rho, dim1=-1, dim2=-2).data.fill_(0.)
        return grad_mean, grad_std, grad_rho

# allow setting Tref
class MnnActivateTrio2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mean, std, rho, t_ref) -> Any:
        ctx.t_ref = t_ref
        loc, dtype = mean.device, mean.dtype
        clone_mean, clone_std = _batch_detach(mean, std)
        mean_out, std_out, chi = mnn_core_func.fast_forward(clone_mean, clone_std, t_ref)
        mean_out, std_out, chi = _to_tensor(loc, dtype, mean_out, std_out, chi)
        ctx.save_for_backward(mean, std, rho, mean_out, std_out, chi)
        rho_out = torch.mul(rho, torch.matmul(chi.unsqueeze(-1), chi.unsqueeze(-2)))
        torch.diagonal(rho_out, dim1=-1, dim2=-2).data.fill_(1.)
        return mean_out, std_out, rho_out

    @staticmethod
    def backward(ctx, grad_output1, grad_output2, grad_output3) -> Any:
        mean_grad, std_grad, rho_grad = grad_output1, grad_output2, grad_output3
        loc, dtype = mean_grad.device, mean_grad.dtype
        mean, std, rho, mean_out, std_out, chi = ctx.saved_tensors
        mean, std, mean_out, std_out, chi = _batch_detach(mean, std, mean_out, std_out, chi)
        mean_grad_mean, mean_grad_std, std_grad_mean, std_grad_std, chi_grad_mean, chi_grad_std = mnn_core_func.fast_backward(mean, std, mean_out, std_out, chi)

        mean_grad_mean, mean_grad_std, std_grad_mean, std_grad_std, chi_grad_mean, chi_grad_std, chi = \
            _to_tensor(loc, dtype, mean_grad_mean, mean_grad_std, std_grad_mean, std_grad_std, chi_grad_mean, chi_grad_std, chi)

        chi_grad_mean = torch.clamp(chi_grad_mean, -1, 1)
        chi_grad_std = torch.clamp(chi_grad_std, -1, 1)
        torch.diagonal(rho, dim1=-1, dim2=-2).data.fill_(0.)
        temp_rho_grad = torch.mul(rho_grad, rho)
        temp_rho_grad = torch.matmul(chi.unsqueeze(-2), temp_rho_grad).squeeze(-2) * 2
        rho_grad_mean = temp_rho_grad * chi_grad_mean
        rho_grad_std = temp_rho_grad * chi_grad_std

        grad_mean = mean_grad * mean_grad_mean + std_grad * std_grad_mean + rho_grad_mean
        grad_std = mean_grad * mean_grad_std + std_grad * std_grad_std + rho_grad_std
        grad_rho = torch.mul(torch.matmul(chi.unsqueeze(-1), chi.unsqueeze(-2)), rho_grad)
        torch.diagonal(grad_rho, dim1=-1, dim2=-2).data.fill_(0.)
        return grad_mean, grad_std, grad_rho, None  

class MnnActivationNoRho(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        mean, std = args
        loc, dtype = mean.device, mean.dtype
        clone_mean, clone_std = _batch_detach(mean, std)
        mean_out = mnn_core_func.forward_fast_mean(clone_mean, clone_std)
        std_out = mnn_core_func.forward_fast_std(clone_mean, clone_std, mean_out)
        clone_mean, clone_std, mean_out, std_out = _to_tensor(loc, dtype, clone_mean, clone_std, mean_out, std_out)
        ctx.save_for_backward(clone_mean, clone_std, mean_out, std_out)
        return mean_out, std_out

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        mean_grad, std_grad = grad_outputs
        loc, dtype = mean_grad.device, mean_grad.dtype
        mean, std, mean_out, std_out = ctx.saved_tensors
        mean, std, mean_out, std_out = _batch_detach(mean, std, mean_out, std_out)
        mean_grad_mean, mean_grad_std = mnn_core_func.backward_fast_mean(mean, std, mean_out)
        std_grad_mean, std_grad_std = mnn_core_func.backward_fast_std(mean, std, mean_out, std_out)
        mean_grad_mean, mean_grad_std, std_grad_mean, std_grad_std = \
            _to_tensor(loc, dtype, mean_grad_mean, mean_grad_std, std_grad_mean, std_grad_std)

        grad_mean = mean_grad * mean_grad_mean + std_grad * std_grad_mean
        grad_std = mean_grad * mean_grad_std + std_grad * std_grad_std
        return grad_mean, grad_std

# allow setting Tref
class MnnActivationNoRho2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mean, std, t_ref) -> Any:
        loc, dtype = mean.device, mean.dtype
        clone_mean, clone_std = _batch_detach(mean, std)
        mean_out = mnn_core_func.forward_fast_mean(clone_mean, clone_std, t_ref)
        std_out = mnn_core_func.forward_fast_std(clone_mean, clone_std, mean_out)
        clone_mean, clone_std, mean_out, std_out = _to_tensor(loc, dtype, clone_mean, clone_std, mean_out, std_out)
        ctx.save_for_backward(clone_mean, clone_std, mean_out, std_out)
        return mean_out, std_out

    @staticmethod
    def backward(ctx, grad_output1, grad_output2) -> Any:
        mean_grad, std_grad = grad_output1, grad_output2
        loc, dtype = mean_grad.device, mean_grad.dtype
        mean, std, mean_out, std_out = ctx.saved_tensors
        mean, std, mean_out, std_out = _batch_detach(mean, std, mean_out, std_out)
        mean_grad_mean, mean_grad_std = mnn_core_func.backward_fast_mean(mean, std, mean_out)
        std_grad_mean, std_grad_std = mnn_core_func.backward_fast_std(mean, std, mean_out, std_out)
        mean_grad_mean, mean_grad_std, std_grad_mean, std_grad_std = \
            _to_tensor(loc, dtype, mean_grad_mean, mean_grad_std, std_grad_mean, std_grad_std)

        grad_mean = mean_grad * mean_grad_mean + std_grad * std_grad_mean
        grad_std = mean_grad * mean_grad_std + std_grad * std_grad_std
        return grad_mean, grad_std, None

def get_core_attr(key: str):
    global mnn_core_func
    return getattr(mnn_core_func, key)


def set_core_attr(key, value):
    global mnn_core_func
    setattr(mnn_core_func, key, value)

def reset_core_attr():
    global mnn_core_func
    mnn_core_func.reset_params()

mnn_activate_trio = MnnActivateTrio.apply
mnn_activate_no_rho = MnnActivationNoRho.apply