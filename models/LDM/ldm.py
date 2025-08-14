#!/usr/bin/python
# -*- coding:utf-8 -*-
import enum

import torch
import torch.nn as nn
from torch_scatter import scatter_sum

import utils.register as R
from utils.oom_decorator import oom_decorator
from utils.nn_utils import length_to_batch_id
from data.format import VOCAB

from .diffusion.dpm_full import FullDPM
from .energies.dist import dist_energy
from ..autoencoder.model import AutoEncoder


@R.register('LDMPepDesign')
class LDMPepDesign(nn.Module):

    def __init__(
            self,
            autoencoder_ckpt,
            autoencoder_no_randomness,
            hidden_size,
            num_steps,
            n_layers,
            dist_rbf=0,
            n_rbf=0,
            cutoff=1.0,
            max_gen_position=30,
            mode='codesign',
            diffusion_opt={}):
        super().__init__()
        self.autoencoder_no_randomness = autoencoder_no_randomness
        self.latent_idx = VOCAB.symbol_to_idx(VOCAB.LAT)

        self.autoencoder: AutoEncoder = torch.load(autoencoder_ckpt, map_location='cpu')
        for param in self.autoencoder.parameters():
            param.requires_grad = False
        self.autoencoder.eval()
        
        self.train_sequence, self.train_structure = True, True
        if mode == 'fixbb':
            self.train_structure = False
        elif mode == 'fixseq':
            self.train_sequence = False
        
        latent_size = self.autoencoder.latent_size if self.train_sequence else hidden_size

        self.abs_position_encoding = nn.Embedding(max_gen_position, latent_size)
        self.diffusion = FullDPM(
            latent_size=latent_size,
            hidden_size=hidden_size,
            n_channel=self.autoencoder.n_channel,
            num_steps=num_steps,
            n_layers=n_layers,
            n_rbf=n_rbf,
            cutoff=cutoff,
            dist_rbf=dist_rbf,
            **diffusion_opt
        )
        if self.train_sequence:
            self.hidden2latent = nn.Linear(hidden_size, self.autoencoder.latent_size)
            self.h_loss_weight = self.autoencoder.latent_n_channel * 3 / self.autoencoder.latent_size  # make loss_X and loss_H about the same size
        if self.train_structure:
            # for better constrained sampling
            self.consec_dist_mean, self.consec_dist_std = None, None

    @oom_decorator
    def forward(self, X, S, mask, position_ids, lengths, atom_mask, L=None):
        '''
            L: [bs, 3, 3], cholesky decomposition of the covariance matrix \Sigma = LL^T
        '''

        # encode latent_H_0 (N*d) and latent_X_0 (N*3)
        with torch.no_grad():
            self.autoencoder.eval()
            H, Z, _, _ = self.autoencoder.encode(X, S, mask, position_ids, lengths, atom_mask, no_randomness=self.autoencoder_no_randomness)

        # diffusion model
        if self.train_sequence:
            S_ori = S.clone()
            S = S.clone()
            S[mask] = self.latent_idx

        with torch.no_grad():
            H_0, (atom_embeddings, _) = self.autoencoder.aa_feature(S, position_ids)
        # position_embedding = self.autoencoder.aa_feature.aa_embedding.res_pos_embedding(position_ids)
        position_embedding = self.abs_position_encoding(torch.where(mask, position_ids + 1, torch.zeros_like(position_ids)))

        if self.train_sequence:
            H_0 = self.hidden2latent(H_0)
            H_0 = H_0.clone()
            H_0[mask] = H
        
        if self.train_structure:
            X = X.clone()
            X[mask] = self.autoencoder._fill_latent_channels(Z)
            # X[mask] = Z.unsqueeze(1).expand_as(X[mask])
            atom_mask = atom_mask.clone()
            atom_mask_gen = atom_mask[mask]
            atom_mask_gen[:, :self.autoencoder.latent_n_channel] = 1
            atom_mask_gen[:, self.autoencoder.latent_n_channel:] = 0
            atom_mask[mask] = atom_mask_gen
            del atom_mask_gen
        else:  # fixbb, only retain backbone atoms in masked region
            atom_mask = self.autoencoder._remove_sidechain_atom_mask(atom_mask, mask)

        loss_dict = self.diffusion.forward(
            H_0=H_0,
            X_0=X,
            S=S_ori,
            position_embedding=position_embedding,
            mask_generate=mask,
            lengths=lengths,
            atom_embeddings=atom_embeddings,
            atom_mask=atom_mask,
            L=L,
            sample_structure=self.train_structure,
            sample_sequence=self.train_sequence
        )

        # loss
        loss = 0
        if self.train_sequence:
            loss = loss + loss_dict['H'] * self.h_loss_weight
        if self.train_structure:
            loss = loss + loss_dict['X']

        return loss, loss_dict

    def set_consec_dist(self, mean: float, std: float):
        self.consec_dist_mean = mean
        self.consec_dist_std = std

    def latent_geometry_guidance(self, H, X, mask_generate, batch_ids, t, tolerance=3, **kwargs):
        assert self.consec_dist_mean is not None and self.consec_dist_std is not None, \
               'Please run set_consec_dist(self, mean, std) to setup guidance parameters'
        if hasattr(self, 'external_energy'):
            base_energy = self.external_energy(t, H, X, mask_generate, batch_ids, **kwargs)
        else:
            base_energy = 0
        return dist_energy(
            X, mask_generate, batch_ids,
            self.consec_dist_mean, self.consec_dist_std,
            tolerance=tolerance, **kwargs
        ) + base_energy

    @torch.no_grad()
    def sample(
        self,
        X, S, mask, position_ids, lengths, atom_mask, L=None,
        sample_opt={
            'pbar': False,
            'energy_func': None,
            'energy_lambda': 0.0,
            'confidence': False
        },
        return_tensor=False,
        optimize_sidechain=True,
        idealize=False,
        guide_mask=None, # to fix some part of the ligand
        cal_likelihood=False
    ):
        use_confidence = sample_opt.pop('confidence', False)
        self.autoencoder.eval()
        if guide_mask is not None:
            with torch.no_grad():
                ref_mask = (~mask) | guide_mask
                ref_batch_ids = length_to_batch_id(S, lengths)[ref_mask]
                ref_lengths = scatter_sum(torch.ones_like(ref_batch_ids), ref_batch_ids, dim=0)
                guide_H, guide_Z, _, _ = self.autoencoder.encode(
                    X[ref_mask], S[ref_mask], mask[ref_mask], position_ids[ref_mask],
                    ref_lengths, atom_mask[ref_mask], no_randomness=self.autoencoder_no_randomness)

        # diffusion sample
        if self.train_sequence:
            S = S.clone()
            if guide_mask is None: S[mask] = self.latent_idx
            else: S[mask & (~guide_mask)] = self.latent_idx

        H_0, (atom_embeddings, _) = self.autoencoder.aa_feature(S, position_ids)
        # position_embedding = self.autoencoder.aa_feature.aa_embedding.res_pos_embedding(position_ids)
        position_embedding = self.abs_position_encoding(torch.where(mask, position_ids + 1, torch.zeros_like(position_ids)))

        if self.train_sequence:
            H_0 = self.hidden2latent(H_0)
            H_0 = H_0.clone()
            if guide_mask is None: H_0[mask] = 0 # no possibility for leakage
            else:
                H_0[guide_mask] = guide_H
                H_0[mask & (~guide_mask)] = 0

        if self.train_structure:
            X = X.clone()
            if guide_mask is None: X[mask] = 0 # no possibility for leakage
            else:
                X[guide_mask] = self.autoencoder._fill_latent_channels(guide_Z)
                X[mask & (~guide_mask)] = 0
            atom_mask = atom_mask.clone()
            atom_mask_gen = atom_mask[mask]
            atom_mask_gen[:, :self.autoencoder.latent_n_channel] = 1
            atom_mask_gen[:, self.autoencoder.latent_n_channel:] = 0
            atom_mask[mask] = atom_mask_gen
            del atom_mask_gen
        else:  # fixbb, only retain backbone atoms in masked region
            atom_mask = self.autoencoder._remove_sidechain_atom_mask(atom_mask, mask)

        sample_opt['sample_sequence'] = self.train_sequence
        sample_opt['sample_structure'] = self.train_structure

        if 'energy_func' in sample_opt:
            if sample_opt['energy_func'] is None:
                pass
            elif sample_opt['energy_func'] == 'default':
                sample_opt['energy_func'] = self.latent_geometry_guidance
            # otherwise this should be a function

        traj = self.diffusion.sample(H_0, X, position_embedding, mask, lengths, atom_embeddings, atom_mask, L, guide_mask=guide_mask, **sample_opt)

        if cal_likelihood:    # by default calculate likelihood
            with torch.set_grad_enabled(True):
                ddim_likelihood = self.cal_sample_likelihood(
                    traj=traj,
                    position_embedding=position_embedding,
                    mask=mask,
                    lengths=lengths,
                    atom_embeddings=atom_embeddings, 
                    atom_mask=atom_mask,
                    L=L,
                    sample_opt=sample_opt
                )

        X_0, H_0 = traj[0]
        if use_confidence:
            batch_confidence = self.diffusion.cal_confidence(
                H_0, X_0, position_embedding, mask, lengths, atom_embeddings,
                atom_mask, L, self.train_structure, self.train_sequence
            )
        X_0, H_0 = X_0[mask][:, :self.autoencoder.latent_n_channel], H_0[mask]

        # autodecoder decode
        batch_X, batch_S, batch_ppls = self.autoencoder.test(
            X, S, mask, position_ids, lengths, atom_mask,
            given_laten_H=H_0, given_latent_X=X_0, return_tensor=return_tensor,
            allow_unk=False, optimize_sidechain=optimize_sidechain, idealize=idealize
        )

        if use_confidence:
            return batch_X, batch_S, batch_confidence.tolist()
        elif cal_likelihood:
            return batch_X, batch_S, ddim_likelihood
        else:
            return batch_X, batch_S, batch_ppls


    @torch.no_grad()
    def cal_confidence(self, X, S, mask, position_ids, lengths, atom_mask, L=None):
        '''
            L: [bs, 3, 3], cholesky decomposition of the covariance matrix \Sigma = LL^T
        '''

        # encode latent_H_0 (N*d) and latent_X_0 (N*3)
        with torch.no_grad():
            self.autoencoder.eval()
            H, Z, _, _ = self.autoencoder.encode(X, S, mask, position_ids, lengths, atom_mask, no_randomness=self.autoencoder_no_randomness)

        # diffusion model
        if self.train_sequence:
            S = S.clone()
            S[mask] = self.latent_idx

        with torch.no_grad():
            H_0, (atom_embeddings, _) = self.autoencoder.aa_feature(S, position_ids)
        # position_embedding = self.autoencoder.aa_feature.aa_embedding.res_pos_embedding(position_ids)
        position_embedding = self.abs_position_encoding(torch.where(mask, position_ids + 1, torch.zeros_like(position_ids)))

        if self.train_sequence:
            H_0 = self.hidden2latent(H_0)
            H_0 = H_0.clone()
            H_0[mask] = H
        
        if self.train_structure:
            X = X.clone()
            X[mask] = self.autoencoder._fill_latent_channels(Z)
            atom_mask = atom_mask.clone()
            atom_mask_gen = atom_mask[mask]
            atom_mask_gen[:, :self.autoencoder.latent_n_channel] = 1
            atom_mask_gen[:, self.autoencoder.latent_n_channel:] = 0
            atom_mask[mask] = atom_mask_gen
            del atom_mask_gen
        else:  # fixbb, only retain backbone atoms in masked region
            atom_mask = self.autoencoder._remove_sidechain_atom_mask(atom_mask, mask)

        return self.diffusion.cal_confidence(
            H_0=H_0,
            X_0=X,
            position_embedding=position_embedding,
            mask_generate=mask,
            lengths=lengths,
            atom_embeddings=atom_embeddings,
            atom_mask=atom_mask,
            L=L,
            sample_structure=self.train_structure,
            sample_sequence=self.train_sequence
        )
    
    def cal_sample_likelihood(
        self,
        traj,
        position_embedding,
        mask,
        lengths,
        atom_embeddings, 
        atom_mask,
        L,
        sample_opt,
    ):
        starts = torch.where(mask & ~mask.roll(1))[0]  
        ends = torch.where(mask & ~mask.roll(-1))[0] 
        delta_log_prob_delta = torch.tensor([0.0 for i in range(len(starts))])
        delta_log_prob_delta = delta_log_prob_delta.to(mask.device)

        for t in traj.keys():
            if t == 0:
                continue
            ## 2.1 get model pred
            x_t, h_t = traj[t]
            x_t = x_t.to(mask.device)
            h_t = h_t.to(mask.device)
            x_t.requires_grad_(True)  
            h_t.requires_grad_(True)  
            
            eps_H, eps_X = self.diffusion.perform_single_step_diffusion(
                h_t=h_t, 
                x_t=x_t, 
                position_embedding=position_embedding,
                t=t, 
                mask_generate=mask,
                lengths=lengths,
                atom_embeddings=atom_embeddings, 
                atom_mask=atom_mask,
                L=L,
                **sample_opt
            )
            assert eps_H.requires_grad and eps_X.requires_grad, "Diffusion output has no gradients!"

            ## 2.2 get the output of score fn, i.e  \nabla_x log p_t(x_t)
            ## pred_type=eps
            vs = self.diffusion.trans_h.var_sched
            alpha = vs.alphas[t].clamp_min(
                vs.alphas[-2]
            )
            alpha_bar = vs.alpha_bars[t]
            alpha_bar_pre = vs.alpha_bars[t-1]
            sigma = vs.sigmas[t]
            
            xt_u_value = torch.sqrt(alpha_bar) * eps_X 
            ht_u_value = torch.sqrt(alpha_bar) * eps_H 
            ## 2.4 calculate div u_value
            sample_times = 5
            
            for i in range(sample_times):
                z_x = torch.randn_like(x_t)
                z_x[~mask, :] = 0
                z_h = torch.randn_like(h_t)
                z_h[~mask, :] = 0
                 # print(z_x.shape, z_h.shape) torch.Size([36, 14, 3]) torch.Size([36, 8])
    
                sum_output = torch.sum(ht_u_value * z_h) + torch.sum(xt_u_value * z_x) 
                sum_output.backward(retain_graph=True)

                grad_x = x_t.grad 
                div_value_x = torch.mean(grad_x * z_x, dim=tuple(range(1, x_t.dim())))  
                grad_h = h_t.grad 
                div_value_h = torch.mean(grad_h * z_h, dim=tuple(range(1, h_t.dim()))) 
                for kk, (start, end) in enumerate(zip(starts, ends)):
                    x_segment_sum = div_value_x[start:end+1].sum(0)  
                    h_segment_sum = div_value_h[start:end+1].sum(0)  
                    delta_log_prob_delta[kk] += (torch.sum(x_segment_sum + h_segment_sum).item()) / sample_times 
                
        delta_log_prob_delta = [item.item() for item in delta_log_prob_delta] 
        return tuple(delta_log_prob_delta)