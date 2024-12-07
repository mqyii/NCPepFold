import os
import torch
import torch.nn as nn
from torch import einsum
from Bio import SeqIO 
from pathlib import Path
from dataclasses import asdict
import numpy as np


# ideal N, CA, C initial coordinates
init_N = torch.tensor([-0.5272, 1.3593, 0.000]).float()
init_CA = torch.zeros_like(init_N)
init_C = torch.tensor([1.5233, 0.000, 0.000]).float()
INIT_CRDS = torch.zeros((27, 3)).float()
INIT_CRDS[:3] = torch.stack((init_N, init_CA, init_C), dim=0) # (3,3)

norm_N = init_N / (torch.norm(init_N, dim=-1, keepdim=True) + 1e-5)
norm_C = init_C / (torch.norm(init_C, dim=-1, keepdim=True) + 1e-5)
cos_ideal_NCAC = torch.sum(norm_N*norm_C, dim=-1) # cosine of ideal N-CA-C bond angle


def rigid_from_3_points(N, Ca, C, non_ideal=False, eps=1e-8):
    #N, Ca, C - [B,L, 3]
    #R - [B,L, 3, 3], det(R)=1, inv(R) = R.T, R is a rotation matrix
    B,L = N.shape[:2]
    
    v1 = C-Ca
    v2 = N-Ca
    e1 = v1/(torch.norm(v1, dim=-1, keepdim=True)+eps)
    u2 = v2-(torch.einsum('bli, bli -> bl', e1, v2)[...,None]*e1)
    e2 = u2/(torch.norm(u2, dim=-1, keepdim=True)+eps)
    e3 = torch.cross(e1, e2, dim=-1)
    R = torch.cat([e1[...,None], e2[...,None], e3[...,None]], axis=-1) #[B,L,3,3] - rotation matrix
    
    if non_ideal:
        v2 = v2/(torch.norm(v2, dim=-1, keepdim=True)+eps)
        cosref = torch.clamp( torch.sum(e1*v2, dim=-1), min=-1.0, max=1.0) # cosine of current N-CA-C bond angle
        costgt = cos_ideal_NCAC.item()
        cos2del = torch.clamp( cosref*costgt + torch.sqrt((1-cosref*cosref)*(1-costgt*costgt)+eps), min=-1.0, max=1.0 )
        cosdel = torch.sqrt(0.5*(1+cos2del)+eps)
        sindel = torch.sign(costgt-cosref) * torch.sqrt(1-0.5*(1+cos2del)+eps)
        Rp = torch.eye(3, device=N.device).repeat(B,L,1,1)
        Rp[:,:,0,0] = cosdel
        Rp[:,:,0,1] = -sindel
        Rp[:,:,1,0] = sindel
        Rp[:,:,1,1] = cosdel
    
        R = torch.einsum('blij,bljk->blik', R,Rp)

    return R, Ca

def get_t(N, Ca, C, non_ideal=False, eps=1e-5):
    I,B,L=N.shape[:3]
    Rs,Ts = rigid_from_3_points(N.view(I*B,L,3), Ca.view(I*B,L,3), C.view(I*B,L,3), non_ideal=non_ideal, eps=eps)
    # Rs: (B,L,3,3)
    Rs = Rs.view(I,B,L,3,3)
    Ts = Ts.view(I,B,L,3)
    t = Ts[:,:,None] - Ts[:,:,:,None] # t[0,1] = residue 0 -> residue 1 vector
    return einsum('iblkj, iblmk -> iblmj', Rs, t) # (I,B,L,L,3)

def get_t_all_atom(all, N, Ca, C, non_ideal=False, eps=1e-5):
    I,B,L=N.shape[:3]
    Rs,Ts = rigid_from_3_points(N.view(I*B,L,3), Ca.view(I*B,L,3), C.view(I*B,L,3), non_ideal=non_ideal, eps=eps)
    Rs = Rs.view(I,B,L,3,3)
    Ts = Ts.view(I,B,L,3)
    Ts = Ts.unsqueeze(3) # (I,B,L,1,3)
    t = all - Ts  # all: (I,B,L,M,3)  t: (I,B,L,M,3)
    # t = Ts[:,:,None] - Ts[:,:,:,None] # t[0,1] = residue 0 -> residue 1 vector
    return einsum('iblkj, iblmk -> iblmj', Rs, t) # (I,B,L,M,3)

class RFAALoss(nn.Module):
    def __init__(self):
        super(RFAALoss, self).__init__()
    
    def forward(self, pred, true, mask_2d, same_chain, negative=False, d_clamp=10.0, d_clamp_inter=30.0, A=10.0, gamma=1.0, eps=1e-6, device="cuda:0"):
        '''
        Calculate Backbone FAPE loss
        Input:
            - pred: predicted coordinates (I, B, L, n_atom, 3)
            - true: true coordinates (B, L, n_atom, 3)
        Output: str loss
        '''
        pred = pred.to(device)
        true = true.to(device)
        mask_2d = mask_2d.to(device)
        same_chain = same_chain.to(device)
        
        I = pred.shape[0]
        true = true.unsqueeze(0)
        # print(true.shape,pred.shape)
        # exit()
        t_tilde_ij = get_t(true[:,:,:,0], true[:,:,:,1], true[:,:,:,2], non_ideal=True)
        t_ij = get_t(pred[:,:,:,0], pred[:,:,:,1], pred[:,:,:,2])
        
        difference = torch.sqrt(torch.square(t_tilde_ij-t_ij).sum(dim=-1) + eps)
        # eij_label = difference[-1].clone().detach()
        
        if d_clamp != None:
            clamp = torch.where(same_chain.bool(), d_clamp, d_clamp_inter)
            clamp = clamp[None]
            difference = torch.clamp(difference, max=clamp)
        # loss = difference / A # (I, B, L, L)
        loss = difference # (I, B, L, L) # A可以调整

        # Get a mask information (ignore missing residue + inter-chain residues)
        # for positive cases, mask = mask_2d
        # for negative cases (non-interacting pairs) mask = mask_2d*same_chain
        if negative:
            mask = mask_2d * same_chain
        else:
            mask = mask_2d
        # calculate masked loss (ignore missing regions when calculate loss)
        loss = (mask[None]*loss).sum(dim=(1,2,3)) / (mask.sum()+eps) # (I)

        # weighting loss
        w_loss = torch.pow(torch.full((I,), gamma, device=pred.device), torch.arange(I, device=pred.device))
        w_loss = torch.flip(w_loss, (0,))
        w_loss = w_loss / w_loss.sum()

        tot_loss = (w_loss * loss).sum()
        return tot_loss


def calc_str_loss(pred, true, mask_2d, same_chain, negative=False, d_clamp=10.0, d_clamp_inter=30.0, A=10.0, gamma=1.0, eps=1e-6, device="cuda:0"):
    '''
    Calculate Backbone FAPE loss
    Input:
        - pred: predicted coordinates (I, B, L, n_atom, 3)
        - true: true coordinates (B, L, n_atom, 3)
    Output: str loss
    '''
    pred = pred.to(device)
    true = true.to(device)
    mask_2d = mask_2d.to(device)
    same_chain = same_chain.to(device)
    
    I = pred.shape[0]
    true = true.unsqueeze(0)
    t_tilde_ij = get_t(true[:,:,:,0], true[:,:,:,1], true[:,:,:,2], non_ideal=True)
    t_ij = get_t(pred[:,:,:,0], pred[:,:,:,1], pred[:,:,:,2])
    
    difference = torch.sqrt(torch.square(t_tilde_ij-t_ij).sum(dim=-1) + eps)
    # eij_label = difference[-1].clone().detach()
    
    if d_clamp != None:
        clamp = torch.where(same_chain.bool(), d_clamp, d_clamp_inter)
        clamp = clamp[None]
        difference = torch.clamp(difference, max=clamp)
    loss = difference / A # (I, B, L, L)

    # Get a mask information (ignore missing residue + inter-chain residues)
    # for positive cases, mask = mask_2d
    # for negative cases (non-interacting pairs) mask = mask_2d*same_chain
    if negative:
        mask = mask_2d * same_chain
    else:
        mask = mask_2d
    # calculate masked loss (ignore missing regions when calculate loss)
    loss = (mask[None]*loss).sum(dim=(1,2,3)) / (mask.sum()+eps) # (I)

    # weighting loss
    w_loss = torch.pow(torch.full((I,), gamma, device=pred.device), torch.arange(I, device=pred.device))
    w_loss = torch.flip(w_loss, (0,))
    w_loss = w_loss / w_loss.sum()

    tot_loss = (w_loss * loss).sum()
    return tot_loss

class RFAALoss_All_Atom(nn.Module):
    def __init__(self):
        super(RFAALoss, self).__init__()
    
    def forward(self, pred, true, mask_2d, same_chain, negative=False, d_clamp=10.0, d_clamp_inter=30.0, A=10.0, gamma=1.0, eps=1e-6, device="cuda:0"):
        '''
        Calculate All-atom FAPE loss
        Input:
            - pred: predicted coordinates (I, B, L,P n_atom, 3)
            - true: true coordinates (B, L, n_atom, 3)
        Intermediate variable:
            - t: Local coordinate. The displacement vector from each residue Ca atom to all heavy atoms (I, B, L, n_atom, 3)
            - t_tilde_ij: (I, B, L, n_atom, 3)
            - t_ij: (I, B, L, n_atom, 3)
        Output: str loss
        '''
        pred = pred.to(device)
        true = true.to(device)
        mask_2d = mask_2d.to(device)
        same_chain = same_chain.to(device)
        
        I = pred.shape[0]
        true = true.unsqueeze(0)
        # print(true.shape,pred.shape)
        # exit()
        t_tilde_ij = get_t_all_atom(true[:,:,:,0], true[:,:,:,1], true[:,:,:,2], non_ideal=True) # (I,B,L,M,3)
        t_ij = get_t_all_atom(pred[:,:,:,0], pred[:,:,:,1], pred[:,:,:,2]) # (I,B,L,M,3)
        
        difference = torch.sqrt(torch.square(t_tilde_ij-t_ij).sum(dim=-1) + eps)
        # eij_label = difference[-1].clone().detach()
        
        if d_clamp != None:
            clamp = torch.where(same_chain.bool(), d_clamp, d_clamp_inter)
            clamp = clamp[None]
            difference = torch.clamp(difference, max=clamp)
        # loss = difference / A # (I, B, L, L)
        loss = difference # (I, B, L, L) # A可以调整

        # Get a mask information (ignore missing residue + inter-chain residues)
        # for positive cases, mask = mask_2d
        # for negative cases (non-interacting pairs) mask = mask_2d*same_chain
        if negative:
            mask = mask_2d * same_chain
        else:
            mask = mask_2d
        # calculate masked loss (ignore missing regions when calculate loss)
        loss = (mask[None]*loss).sum(dim=(1,2,3)) / (mask.sum()+eps) # (I)

        # weighting loss
        w_loss = torch.pow(torch.full((I,), gamma, device=pred.device), torch.arange(I, device=pred.device))
        w_loss = torch.flip(w_loss, (0,))
        w_loss = w_loss / w_loss.sum()

        tot_loss = (w_loss * loss).sum()
        return tot_loss   