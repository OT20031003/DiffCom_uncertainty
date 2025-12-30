import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import utils_model

__CONDITIONING_METHOD__ = {}


def register_conditioning_method(name: str):
    def wrapper(cls):
        if __CONDITIONING_METHOD__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __CONDITIONING_METHOD__[name] = cls
        return cls

    return wrapper


def get_conditioning_method(name: str, **kwargs):
    if __CONDITIONING_METHOD__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __CONDITIONING_METHOD__[name](**kwargs)


class ConsistencyLoss(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        zeta = config.diffcom_series[config.conditioning_method]['zeta']
        gamma = config.diffcom_series[config.conditioning_method]['gamma']
        self.weight = {
            'x_mse': gamma,
            'ofdm_sig': zeta,
        }
        self.mask_info = None  # {key: (mask, weight)}

    def set_mask(self, mask_dict, weight_val):
        """
        Configure masks for Pass 2 retransmission.
        mask_dict: {'ofdm_sig': latent_mask, 'x_mse': pixel_mask}
        weight_val: scalar float (e.g., 5.0)
        """
        self.mask_info = {}
        for k, v in mask_dict.items():
            if v is not None:
                self.mask_info[k] = (v, weight_val)

    def forward(self, measurement, x_0_hat, cof, operator, operation_mode):
        x_0_hat = (x_0_hat / 2 + 0.5)  # .clip(0, 1)
        s = operator.encode(x_0_hat)
        if operation_mode == 'latent':
            recon_measurement = {
                'ofdm_sig': operator.forward(s, cof)
            }
        elif operation_mode == 'pixel':
            recon_measurement = {
                'x_mse': x_0_hat
            }
        elif operation_mode == 'joint':
            ofdm_sig = operator.forward(s, cof)
            s_hat = operator.transpose(ofdm_sig, cof)
            x_confirming = operator.decode(s_hat)
            recon_measurement = {
                'ofdm_sig': ofdm_sig,
                'x_mse': x_confirming
            }
        loss = {}
        for key in recon_measurement.keys():
            diff_sq = (measurement[key] - recon_measurement[key]) ** 2
            
            # Apply Mask Weighting if retransmission is active (Strategy A)
            if self.mask_info is not None and key in self.mask_info:
                mask, w_val = self.mask_info[key]
                # If shapes mismatch (e.g. flattened latent), reshape mask or broadcast
                if mask.shape != diff_sq.shape:
                     if key == 'ofdm_sig' and diff_sq.dim() == 2: 
                         # mask: (B, C, H, W) -> flatten -> (B, -1)
                         mask_flat = mask.reshape(diff_sq.shape[0], -1)
                         weight_map = 1.0 + (w_val - 1.0) * mask_flat
                     else:
                         # Try standard broadcasting
                         weight_map = 1.0 + (w_val - 1.0) * mask
                else:
                    weight_map = 1.0 + (w_val - 1.0) * mask
                
                # Weighted MSE
                weighted_diff_sq = diff_sq * weight_map
                # Use sqrt of sum to mimic L2 norm scale but weighted
                loss[key] = self.weight[key] * torch.sqrt(torch.sum(weighted_diff_sq))
            else:
                loss[key] = self.weight[key] * torch.linalg.norm(measurement[key] - recon_measurement[key])
        return loss


def get_lr(config, t, T):
    lr_base = config['learning_rate']
    # exponential decay to 0
    if config['lr_schedule'] == 'exp':
        lr_min = config['lr_min']
        lr = lr_min + (lr_base - lr_min) * np.exp(-t / T)
    # linear decay
    elif config['lr_schedule'] == 'linear':
        lr_min = config['lr_min']
        lr = lr_min + (lr_base - lr_min) * (t / T)
    # constant
    else:
        lr = lr_base
    return lr


@register_conditioning_method(name='diffcom')
class DiffCom(nn.Module):
    def __init__(self):
        super().__init__()
        self.conditioning_method = 'latent'

    def conditioning(self, config, i, ns, x_t, h_t, power,
                     measurement, unet, diffusion, operator, loss_wrapper, last_timestep):
        h_0_hat = h_t
        h_t_minus_1_prime = h_t
        h_t_minus_1 = h_t

        t_step = ns.seq[i]
        sigma_t = ns.reduced_alpha_cumprod[t_step].cpu().numpy()
        x_t = x_t.requires_grad_()
        x_t_minus_1_prime, x_0_hat, _ = utils_model.model_fn(x_t,
                                                             noise_level=sigma_t * 255,
                                                             model_out_type='pred_x_prev_and_start', \
                                                             model_diffusion=unet,
                                                             diffusion=diffusion,
                                                             ddim_sample=config.ddim_sample)
        if last_timestep:
            loss = loss_wrapper.forward(measurement, x_0_hat, h_0_hat, operator, self.conditioning_method)
            return x_0_hat, h_0_hat, x_t_minus_1_prime, h_t_minus_1_prime, loss
        else:
            loss = loss_wrapper.forward(measurement, x_0_hat, h_t, operator, self.conditioning_method)
            total_loss = sum(loss.values())
            x_grad = torch.autograd.grad(outputs=total_loss, inputs=x_t)[0]
            learning_rate = get_lr(config.diffcom_series[config.conditioning_method], t_step,
                                   ns.t_start - 1)
            x_t_minus_1 = x_t_minus_1_prime - x_grad * learning_rate
            x_t_minus_1 = x_t_minus_1.detach_()
            return x_0_hat, h_0_hat, x_t_minus_1, h_t_minus_1, loss


@register_conditioning_method(name='hifi_diffcom')
class HiFiDiffCom(DiffCom):
    def __init__(self):
        super().__init__()
        self.conditioning_method = 'joint'


@register_conditioning_method(name='blind_diffcom')
class BlindDiffCom(DiffCom):
    def __init__(self):
        super().__init__()

    def conditioning(self, config, i, ns, x_t, h_t, power,
                     measurement, unet, diffusion, operator, loss_wrapper, last_timestep):
        t_step = ns.seq[i]
        sigma_t = ns.reduced_alpha_cumprod[t_step].cpu().numpy()
        x_t = x_t.requires_grad_()
        x_t_minus_1_prime, x_0_hat, _ = utils_model.model_fn(x_t,
                                                             noise_level=sigma_t * 255,
                                                             model_out_type='pred_x_prev_and_start', \
                                                             model_diffusion=unet,
                                                             diffusion=diffusion,
                                                             ddim_sample=config.ddim_sample)

        assert (config.conditioning_method == 'blind_diffcom')

        h_t = h_t.requires_grad_()
        h_score = - h_t / (power ** 2)
        h_0_hat = (1 / ns.alphas_cumprod[t_step]) * (
                h_t + ns.sqrt_1m_alphas_cumprod[t_step] * h_score)
        h_t_minus_1_prime = ns.posterior_mean_coef2[t_step] * h_t + ns.posterior_mean_coef1[t_step] * h_0_hat + \
                            ns.posterior_variance[t_step] * (torch.randn_like(h_t) + 1j * torch.randn_like(h_t))

        if last_timestep:
            loss = loss_wrapper.forward(measurement, x_0_hat, h_0_hat, operator, self.conditioning_method)
            return x_0_hat, h_0_hat, x_t_minus_1_prime, h_t_minus_1_prime, loss
        else:
            loss = loss_wrapper.forward(measurement, x_0_hat, h_0_hat, operator, self.conditioning_method)
            total_loss = sum(loss.values())
            x_grad, h_t_grad = torch.autograd.grad(outputs=total_loss, inputs=[x_t, h_t])
            learning_rate = config.diffcom_series['blind_diffcom']['learning_rate']
            learning_rate = (learning_rate - 0) * (t_step / (ns.t_start - 1))
            x_t_minus_1 = x_t_minus_1_prime - x_grad * learning_rate
            x_t_minus_1 = x_t_minus_1.detach_()
            lr_h = config.diffcom_series['blind_diffcom']['h_lr']
            lr_h = (lr_h - 0) * (t_step / (ns.t_start - 1))
            h_t_minus_1 = h_t_minus_1_prime - h_t_grad * lr_h
            h_t_minus_1 = h_t_minus_1.detach_()
            return x_0_hat, h_0_hat, x_t_minus_1, h_t_minus_1, loss

# ==========================================
# Uncertainty & Retransmission Utils
# ==========================================

def calculate_uncertainty(x0_preds_list):
    """
    Calculate Temporal Variance based on predictions x0_hat collected over timesteps.
    x0_preds_list: List of tensor (B, C, H, W)
    Returns: uncertainty_map (B, 1, H, W)
    """
    if not x0_preds_list:
        return None
    # stack -> (T, B, C, H, W)
    preds_stack = torch.stack(x0_preds_list)
    # variance over time dimension (dim=0)
    variance_map = torch.var(preds_stack, dim=0) # (B, C, H, W)
    # Average over channels for a spatial mask
    uncertainty_map = torch.mean(variance_map, dim=1, keepdim=True) # (B, 1, H, W)
    return uncertainty_map

def create_retransmission_mask(uncertainty_map, rate):
    """
    Create a binary mask for the top `rate` percentile of uncertainty.
    uncertainty_map: (B, 1, H, W)
    rate: float (0.0 - 1.0)
    Returns: binary mask (B, 1, H, W)
    """
    b, c, h, w = uncertainty_map.shape
    mask = torch.zeros_like(uncertainty_map)
    for i in range(b):
        flat = uncertainty_map[i].flatten()
        k = int(flat.numel() * rate)
        if k > 0:
            threshold = torch.kthvalue(flat, flat.numel() - k + 1).values
            mask[i] = (uncertainty_map[i] >= threshold).float()
    return mask