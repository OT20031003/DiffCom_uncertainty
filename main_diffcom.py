import argparse
import logging
import os
import os.path
import random
import shutil
import copy

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import yaml
from tqdm.auto import tqdm

# 必要なモジュールをインポート
from conditioning_method.diffcom import get_conditioning_method, ConsistencyLoss, calculate_uncertainty, create_retransmission_mask
from data.datasets import get_test_loader
from guided_diffusion.measurement import get_operator
from guided_diffusion.noise_schedule import NoiseSchedule
from guided_diffusion.script_util import model_and_diffusion_defaults, create_model_and_diffusion, args_to_dict
from utils.util import Config, MetricWrapper, DictAverageMeter
from utils import util, utils_logger, utils_model


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt", type=str, default='./configs/diffcom.yaml', help="Path to option YMAL file.")
    args = parser.parse_args()
    # Load the YAML file
    with open(args.opt, 'r') as file:
        config = yaml.safe_load(file)
    config = Config(config)
    if config.conditioning_method == 'blind_diffcom':
        # default config for blind_diffcom
        assert config.channel_type == 'ofdm_tdl'
        assert not config.CSNR_adapt_t_start

    cond_config = Config(config.getattr('diffcom_series'))
    conditioning_method = Config(cond_config.getattr(config.conditioning_method))
    config.world_size = torch.cuda.device_count()
    config.opt = args.opt
    config.skip = cond_config.num_train_timesteps // cond_config.iter_num  # skip interval
    config.sigma = np.sqrt(1.0 / (2 * 10 ** (config.CSNR / 10)))  # noise level from channel

    # paths
    config.model_zoo = os.path.join(config.cwd, 'model_zoo')  # fixed
    config.testsets = os.path.join(config.cwd, 'testsets')  # fixed
    config.results = os.path.join(config.cwd, 'results')  # fixed
    config.results = os.path.join(config.results, config.testset_name)
    config.results = os.path.join(config.results, config.conditioning_method)

    if config.operator_name == 'djscc':
        config.results = os.path.join(config.results, config.operator_name + '_{}'.format(config.djscc['channel_num']))
    elif config.operator_name == 'ntscc':
        if config.ntscc['compatible']:
            config.results = os.path.join(config.results, config.operator_name + '_{}_{}'.format(config.ntscc['eta'],
                                                                                                 config.ntscc[
                                                                                                     'qp_level']))
        else:
            config.results = os.path.join(config.results,
                                          config.operator_name + '_plus_{}'.format(config.ntscc['qp_level']))

    config.results = os.path.join(config.results, f'{config.channel_type}_{config.CSNR.__str__().zfill(2)}dB')

    config.result_name = f'zeta{conditioning_method.zeta}'
    config.result_name += f'_seed{config.seed}'
    config.result_name += f'_gamma{conditioning_method.gamma}'
    config.result_name += f'_faststart_N{config.N}' if config.CSNR_adapt_t_start else ''
    
    # Append Retransmission tag to result name if enabled
    if hasattr(config, 'retransmission') and config.retransmission['enable']:
        config.result_name += f"_RETX{config.retransmission['rate']}"

    if config.channel_type == 'ofdm_tdl':
        ofdm_config = Config(config.ofdm_tdl)
        config.result_name += '_BLIND_h_lr{}_'.format(
            conditioning_method.h_lr) if config.conditioning_method == 'blind_diffcom' else f'_{ofdm_config.channel_est}_{ofdm_config.equalization}'
        if ofdm_config.is_clip:
            config.result_name += '_CLIP{}'.format(ofdm_config.clip_ratio)
        if ofdm_config.K < ofdm_config.L:
            config.result_name += f'_ISI'

    config.result_name += f'_NFE{cond_config.iter_num}_{config.model_name}'
    config.model_path = os.path.join(config.model_zoo, config.model_name + '.pt')
    config.testsets_path = os.path.join(config.testsets, config.testset_name)
    config.save_path = os.path.join(config.results, config.result_name)
    util.mkdir(config.save_path)

    # set random seed everywhere
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)  # for multi-GPU.
    np.random.seed(config.seed)  # Numpy module.
    random.seed(config.seed)  # Python random module.
    torch.manual_seed(config.seed)
    return config


def run_diffusion_loop(config, noise_schedule, unet, diffusion, operator, cond_method, 
                       measurement, x_init, cof_init, power, loss_wrapper, names, 
                       input_image, metric_wrapper,
                       save_prefix="", return_history=False):
    """
    Executes the diffusion sampling loop.
    Returns: x_recon, x0_history, L_m_list, L_c_list, H_loss_list
    """
    seq = noise_schedule.seq
    ofdm_config = Config(config.ofdm_tdl) if config.channel_type == 'ofdm_tdl' else None
    
    x_t = x_init
    h_t = cof_init
    
    x0_history = []
    psnr_list = []
    lpips_list = []
    dists_list = []
    L_m_list = []
    H_loss_list = []
    L_c_list = []
    
    pbar = tqdm(range(len(seq)), ncols=140, desc=f"Sampling {save_prefix}")
    x_recon = None
    
    for i in pbar:
        x_0_hat, h_0_hat, x_t, h_t, norm, noise = cond_method(config, i, noise_schedule,
                                                       x_init if i == 0 else x_t,
                                                       cof_init if i == 0 else h_t,
                                                       power if config.conditioning_method == 'blind_diffcom' else None,
                                                       measurement, unet, diffusion, operator, loss_wrapper,
                                                       last_timestep=(seq[i] == seq[-1]))
        print(noise)
        if return_history:
            x0_history.append(x_0_hat.detach().cpu())

        # Save intermediate results
        if (seq[i]) % config.diffcom_series['save_recon_every'] == 0:
            save_path = os.path.join(config.save_path, f"recon_{save_prefix}")
            util.mkdir(save_path)
            util.mkdir(save_path + '/x_0^t')
            torchvision.utils.save_image(x_0_hat / 2 + 0.5,
                                         os.path.join(save_path + '/x_0^t', f"x_0^{seq[i].__str__().zfill(4)}.png"))

            if config.conditioning_method == 'blind_diffcom':
                cof_hat = h_0_hat[0, 0, :ofdm_config.L].cpu().detach().numpy()
                cof_hat_real = cof_hat.real
                cof_hat_imag = cof_hat.imag

                save_cof_path = os.path.join(config.save_path, f"recon_{save_prefix}/{names[0][:-4]}")
                util.mkdir(save_cof_path)
                torchvision.utils.save_image(x_0_hat / 2 + 0.5,
                                             os.path.join(save_cof_path, f"x_0^{seq[i].__str__().zfill(4)}.png"))
                save_cof_path = os.path.join(config.save_path, f"recon_{save_prefix}/{names[0][:-4]}_cof")
                util.mkdir(save_cof_path)
                # plot estimated channel response h_0_hat
                plt.clf()
                plt.figure(figsize=(4, 4))
                font = {'family': 'serif', 'weight': 'normal', 'size': 12}
                matplotlib.rc('font', **font)
                ax = plt.gca()
                BoundWidth = 1.5
                ax.spines['bottom'].set_linewidth(BoundWidth)
                ax.spines['left'].set_linewidth(BoundWidth)
                ax.spines['top'].set_linewidth(BoundWidth)
                ax.spines['right'].set_linewidth(BoundWidth)

                plt.scatter(cof_hat_real, cof_hat_imag, marker='o', s=80, color='c', zorder=1, label=r'$\hat{h}_{0|t}$')
                plt.xlim(-0.6, 0.6)
                plt.ylim(-0.6, 0.6)
                plt.xticks(np.arange(-0.6, 0.7, 0.2))
                plt.yticks(np.arange(-0.6, 0.7, 0.2))
                plt.grid()
                plt.savefig(os.path.join(save_cof_path, f'cof_hat_{seq[i].__str__().zfill(4)}.png'),
                            bbox_inches='tight')
                plt.close()

        # Metrics calculation for monitoring
        metrics = metric_wrapper((x_0_hat / 2 + 0.5).detach(), input_image)
        if i > 100 and metrics['psnr'] < 6:
            print('Failed to converge, Please check the reverse diffusion process.')
            break

        message = {'t_step': seq[i],
                   'H_dist': 0.0,
                   'L_m': norm['ofdm_sig'].item() if 'ofdm_sig' in norm.keys() else 0.0,
                   'L_c': norm['x_mse'].item() if 'x_mse' in norm.keys() else 0.0,
                   'PSNR': metrics['psnr'],
                   'LPIPS': metrics['lpips'],
                   'DISTS': metrics['dists']}
        
        if config.channel_type == 'ofdm_tdl':
             message['H_dist'] = torch.linalg.norm(
                h_t[..., :ofdm_config.L] - measurement["cof_gt"][..., :ofdm_config.L]).item()
             H_loss_list.append(message['H_dist'])
        else:
             H_loss_list.append(0.0)

        L_m_list.append(message['L_m'])
        L_c_list.append(message['L_c'])
        psnr_list.append(metrics['psnr'])
        lpips_list.append(metrics['lpips'])
        dists_list.append(metrics['dists'])
        
        pbar.set_postfix(message, refresh=True)
        
    x_recon = (x_t / 2 + 0.5)
    
    # Plot Loss Curves
    plt.clf()
    font = {'family': 'serif', 'weight': 'normal', 'size': 12}
    matplotlib.rc('font', **font)
    ax = plt.gca()
    BoundWidth = 1.5
    ax.spines['bottom'].set_linewidth(BoundWidth)
    ax.spines['left'].set_linewidth(BoundWidth)
    ax.spines['top'].set_linewidth(BoundWidth)
    ax.spines['right'].set_linewidth(BoundWidth)
    plt.plot(L_m_list)
    plt.xlabel('Timestep')
    plt.ylabel('$\mathcal{L}_m$')
    util.mkdir(config.save_path + '/chart')
    plt.savefig(config.save_path + f'/chart/L_Loss_{names[0][:-4]}_{save_prefix}.png', bbox_inches='tight')
    plt.close()

    if config.conditioning_method == 'blind_diffcom':
        plt.clf()
        font = {'family': 'serif', 'weight': 'normal', 'size': 12}
        matplotlib.rc('font', **font)
        ax = plt.gca()
        BoundWidth = 1.5
        ax.spines['bottom'].set_linewidth(BoundWidth)
        ax.spines['left'].set_linewidth(BoundWidth)
        ax.spines['top'].set_linewidth(BoundWidth)
        ax.spines['right'].set_linewidth(BoundWidth)
        plt.plot(H_loss_list)
        plt.xlabel('Timestep')
        plt.ylabel('$\|\bm{h}^* - \bm{h}_{0|t} \|_2^2$')
        util.mkdir(config.save_path + '/chart')
        plt.savefig(config.save_path + f'/chart/H_Loss_{names[0][:-4]}_{save_prefix}.png', bbox_inches='tight')
        plt.close()

    return x_recon, x0_history, L_m_list, L_c_list, H_loss_list


def p_sample_loop(config, noise_schedule, unet, diffusion, operator, cond_method, dataloader, device, logger):
    logger.info('【Config】: model_name: {}'.format(config.model_name))
    logger.info('【Config】: testset_name: {}'.format(config.testset_name))
    logger.info('【Config】: conditioning_method: {}'.format(config.conditioning_method))
    for key, value in config.diffcom_series[config.conditioning_method].items():
        logger.info('【Config】: {}: {}'.format(key, value))
    logger.info('【Config】: channel_type: {}'.format(config.channel_type))
    logger.info('【Config】: CSNR: {}'.format(config.CSNR))
    ofdm_config = Config(config.ofdm_tdl)
    logger.info('【Config】: {} channel estimation'.format(ofdm_config.channel_est))
    logger.info('【Config】: {} equalization'.format(ofdm_config.equalization))
    logger.info('【Config】: 【BLIND MODE】') if config.conditioning_method == 'blind_diffcom' else None

    metric_wrapper = MetricWrapper().to(device)
    results = DictAverageMeter()
    loss_wrapper = ConsistencyLoss(config, device)

    # Check for Retransmission Config
    enable_retx = False
    if hasattr(config, 'retransmission') and config.retransmission['enable']:
        enable_retx = True
        logger.info(f"【Retransmission】 Enabled. Rate: {config.retransmission['rate']}, Weight: {config.retransmission['guidance_weight']}")

    for idx, batch in enumerate(dataloader):
        input_image, names = batch
        input_image = input_image.to(device)
        config.batch_size = input_image.shape[0]
        
        # --- Observe (Simulate Initial Transmission) ---
        measurement = operator.observe_and_transpose(input_image)
        
        torch.manual_seed(config.seed + 1)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed + 1)
            torch.cuda.manual_seed_all(config.seed + 1)
        np.random.seed(config.seed + 1)
        random.seed(config.seed + 1)
        torch.manual_seed(config.seed + 1)

        if config.channel_type == 'ofdm_tdl' and not (config.conditioning_method == 'blind_diffcom'):
            H_loss_gt = torch.linalg.norm(measurement['cof_est'] - measurement["cof_gt"])
            logger.info(f"batch{idx + 1:->4d}--> 【Init】 H_Loss cof_gt: {H_loss_gt:.4f}")

        # --- Measurement Saving ---
        util.mkdir(config.save_path + '/measurement')
        util.imsave_batch(util.tensor2uint_batch(measurement['x_mse']), names, config.save_path + '/measurement',
                          f"measurement_")

        # --- Baseline Metric ---
        baseline_metric = metric_wrapper(measurement['x_mse'], input_image)
        logger.info(
            f"batch{idx + 1:->4d}--> 【Baseline】"
            f"CBR: {measurement['channel_usage'] / measurement['x_mse'].numel():.4f},"
            f"PSNR: {baseline_metric['psnr']:.2f}dB, "
            f"LPIPS: {baseline_metric['lpips']:.4f}, "
            f"DISTS: {baseline_metric['dists']:.4f}, "
            f"MSSSIM: {baseline_metric['msssim']:.4f}")

        # --- Initialization ---
        x_init = noise_schedule.sqrt_alphas_cumprod[noise_schedule.t_start] * (2 * measurement['x_mse'] - 1) + \
                 noise_schedule.sqrt_1m_alphas_cumprod[
                     noise_schedule.t_start] * torch.randn_like(input_image)
        
        cof_gt = 0 + 0j
        cof_init = measurement['cof_est']
        power = None
        if config.conditioning_method == 'blind_diffcom':
             # Blind Init Logic
             plt.clf()
             plt.figure(figsize=(4, 4))
             font = {'family': 'serif', 'weight': 'normal', 'size': 12}
             matplotlib.rc('font', **font)
             ax = plt.gca()
             BoundWidth = 1.5
             ax.spines['bottom'].set_linewidth(BoundWidth)
             ax.spines['left'].set_linewidth(BoundWidth)
             ax.spines['top'].set_linewidth(BoundWidth)
             ax.spines['right'].set_linewidth(BoundWidth)
             cof_gt = measurement['cof_gt'][0, 0, :ofdm_config.L].cpu().numpy()
             cof_gt_real = cof_gt.real
             cof_gt_imag = cof_gt.imag
             plt.scatter(cof_gt_real, cof_gt_imag, marker='x', color='r', s=80)
             plt.xlim(-0.6, 0.6)
             plt.ylim(-0.6, 0.6)
             plt.xticks(np.arange(-0.6, 0.7, 0.2))
             plt.yticks(np.arange(-0.6, 0.7, 0.2))
             plt.grid()
             util.mkdir(config.save_path + '/chart')
             plt.savefig(config.save_path + '/chart/channel_response.png', bbox_inches='tight')
             plt.close()

             power = torch.exp(-torch.arange(ofdm_config.L).float() / ofdm_config.decay).view(1, 1, ofdm_config.L).to(device)
             power = power / sum(power)
             cof_init_real = torch.randn_like(measurement['cof_gt'][..., :ofdm_config.L]) * power
             cof_init_imag = torch.randn_like(measurement['cof_gt'][..., :ofdm_config.L]) * power
             cof_init = cof_init_real + 1j * cof_init_imag
             cof_init = noise_schedule.sqrt_alphas_cumprod[noise_schedule.t_start] * cof_init + \
                        noise_schedule.sqrt_1m_alphas_cumprod[noise_schedule.t_start] * torch.randn_like(cof_init)

        # =======================================================
        # Pass 1: Initial Transmission
        # =======================================================
        loss_wrapper.set_mask({}, 1.0) # Ensure mask is cleared
        x_recon_1, x0_history, L_m_list_1, L_c_list_1, H_loss_list_1 = run_diffusion_loop(
            config, noise_schedule, unet, diffusion, operator, cond_method,
            measurement, x_init, cof_init, power, loss_wrapper, names, 
            input_image, metric_wrapper,
            save_prefix="Pass1", return_history=True
        )
        
        metrics_1 = metric_wrapper(x_recon_1.detach(), input_image)
        logger.info(
            f"batch{idx + 1:->4d}--> 【Pass 1】 "
            f"PSNR: {metrics_1['psnr']:.2f}dB, LPIPS: {metrics_1['lpips']:.4f}, "
            f"DISTS: {metrics_1['dists']:.4f}, MSSSIM: {metrics_1['msssim']:.4f}")
        
        recon_image_1 = util.tensor2uint_batch(x_recon_1)
        util.imsave_batch(recon_image_1, names, config.save_path + '/recon_Pass1', f"{config.model_name}_")

        if enable_retx:
            # =======================================================
            # Uncertainty & Retransmission Logic
            # =======================================================
            
            # 1. Calculate Uncertainty (Pixel Space)
            unc_map = calculate_uncertainty(x0_history).to(device) # (B, 1, H, W)
            
            # 2. Create Masks
            pixel_mask = create_retransmission_mask(unc_map, config.retransmission['rate'])
            
            # Latent Mask (for ofdm_sig guidance) - Downsample
            # Note: DeepJSCC downsampling factor is 4 (256 -> 64)
            latent_h, latent_w = input_image.shape[2] // 4, input_image.shape[3] // 4
            latent_mask = torch.nn.functional.interpolate(pixel_mask, size=(latent_h, latent_w), mode='nearest')
            latent_c = config.djscc['channel_num'] 
            latent_mask = latent_mask.repeat(1, latent_c, 1, 1) 
            
            # 3. Simulate Retransmission (Update Measurement)
            measurement_retx = operator.observe_and_transpose(input_image)
            
            # Update 'ofdm_sig'
            
            latent_mask_flat = latent_mask.reshape(config.batch_size, -1)
            ofdm_sig_refined = measurement['ofdm_sig'] #* (1 - latent_mask_flat) + measurement_retx['ofdm_sig'] * latent_mask_flat
            # ofdm_sig_refined = measurement['ofdm_sig'] * (1 - latent_mask_flat) + \
            #        (measurement['ofdm_sig'] + measurement_retx['ofdm_sig']) / 2.0 * latent_mask_flat
            # Update 'x_mse'
            s_hat_refined = operator.channel.transpose(ofdm_sig_refined, measurement['cof_est']) 
            x_mse_refined = operator.decode(s_hat_refined)
            
            measurement_refined = copy.deepcopy(measurement)
            measurement_refined['ofdm_sig'] = ofdm_sig_refined
            measurement_refined['x_mse'] = x_mse_refined
            # ★実証用に追加: Pass 1 と同じシードを再設定する
            # これにより、拡散過程の乱数も同じになるため、画像が完全に一致するはずです
            torch.manual_seed(config.seed + 1)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(config.seed + 1)
            # =============== 【追加すべき修正】 ===============
            # 解説: Pass 1 では x_init 生成時に torch.randn_like が呼ばれ、乱数が消費されました。
            # Pass 2 で拡散ループに入る前に、同じ回数だけ乱数を消費して状態を合わせる必要があります。
            # ※ blind_diffcom の場合は cof_init 用の乱数も消費する必要がありますが、
            #   ログを見る限り blind ではないようなので以下だけでOKです。
            _ = torch.randn_like(input_image) 
            # ================================================
            # Save Uncertainty and Mask
            util.mkdir(config.save_path + '/uncertainty')
            util.imsave_batch(util.tensor2uint_batch(unc_map.repeat(1,3,1,1)/unc_map.max()), names, config.save_path + '/uncertainty', f"unc_")
            util.imsave_batch(util.tensor2uint_batch(pixel_mask.repeat(1,3,1,1)), names, config.save_path + '/uncertainty', f"mask_")

            # =======================================================
            # Pass 2: Refined Decoding
            # =======================================================
            loss_wrapper.set_mask({
                'ofdm_sig': latent_mask, 
                'x_mse': pixel_mask      
            }, config.retransmission['guidance_weight'])
            
            x_recon_2, _, L_m_list_2, L_c_list_2, H_loss_list_2 = run_diffusion_loop(
                config, noise_schedule, unet, diffusion, operator, cond_method,
                measurement_refined, x_init, cof_init, power, loss_wrapper, names, 
                input_image, metric_wrapper,
                save_prefix="Pass2", return_history=False
            )
            
            metrics_2 = metric_wrapper(x_recon_2.detach(), input_image)
            logger.info(
                f"batch{idx + 1:->4d}--> 【Pass 2】 "
                f"PSNR: {metrics_2['psnr']:.2f}dB, LPIPS: {metrics_2['lpips']:.4f}, "
                f"DISTS: {metrics_2['dists']:.4f}, MSSSIM: {metrics_2['msssim']:.4f}")
            
            # Use Pass 2 metrics for final result averaging
            final_metric = metrics_2
            final_metric['L_m'] = L_m_list_2[-1]
            final_metric['L_c'] = L_c_list_2[-1]
            final_metric['H_Loss'] = H_loss_list_2[-1]
            
            x_recon = x_recon_2
            recon_image = util.tensor2uint_batch(x_recon)
            util.imsave_batch(recon_image, names, config.save_path + '/recon_Pass2', f"{config.model_name}_")
            
            # Save Visual Comparison (Pass 1 vs Pass 2)
            fig, axs = plt.subplots(1, 4, figsize=(20, 5))
            axs[0].imshow(util.tensor2uint(input_image[0]))
            axs[0].set_title('Ground Truth')
            axs[1].imshow(util.tensor2uint(measurement['x_mse'][0]))
            axs[1].set_title('Measurement (Pass 1)')
            axs[2].imshow(util.tensor2uint(x_recon_1[0]))
            axs[2].set_title(f'Pass 1 ({metrics_1["psnr"]:.2f}dB)')
            axs[3].imshow(util.tensor2uint(x_recon_2[0]))
            axs[3].set_title(f'Pass 2 ({metrics_2["psnr"]:.2f}dB)')
            for ax in axs: ax.axis('off')
            plt.tight_layout()
            plt.savefig(config.save_path + f'/visual_compare_retx_{idx}.png')
            plt.close()

        else:
            # If no retransmission, just use Pass 1 results
            final_metric = metrics_1
            final_metric['L_m'] = L_m_list_1[-1]
            final_metric['L_c'] = L_c_list_1[-1]
            final_metric['H_Loss'] = H_loss_list_1[-1]
            x_recon = x_recon_1
            recon_image = util.tensor2uint_batch(x_recon)
            util.imsave_batch(recon_image, names, config.save_path + '/recon', f"{config.model_name}_")
            
            # Standard Visual Comparison
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            axs[0].imshow(util.tensor2uint(x_recon[0]))
            axs[0].set_title('Reconstructed')
            axs[1].imshow(util.tensor2uint(input_image[0]))
            axs[1].set_title('Ground Truth')
            axs[2].imshow(util.tensor2uint(measurement['x_mse'][0]))
            axs[2].set_title('Measurement')
            for ax in axs: ax.axis('off')
            plt.tight_layout()
            plt.savefig(config.save_path + f'/visual_compare_{idx}.png')
            plt.close()

        # Update averages
        results.update(final_metric)
        
    # --- Final Average Logging ---
    logger.info('-----------> Method: {}'.format(config.conditioning_method))
    logger.info('-----------> Average PSNR: {}'.format(results.avg['psnr']))
    logger.info('-----------> Average LPIPS: {}'.format(results.avg['lpips']))
    logger.info('-----------> Average DISTS: {}'.format(results.avg['dists']))
    logger.info('-----------> Average MSSSIM: {}'.format(results.avg['msssim']))
    if config.conditioning_method == 'blind_diffcom':
        logger.info('-----------> Average H_Loss: {}'.format(results.avg['H_Loss']))
    logger.info('-----------> Results Save to {}'.format(config.save_path))
    return results


def main():
    config = parse_args_and_config()
    device = torch.device('cuda:{}'.format(config.gpu_id) if torch.cuda.is_available() else 'cpu')
    config.device = device

    # set up logger
    logger_name = config.result_name
    utils_logger.logger_info(logger_name, log_path=os.path.join(config.save_path, logger_name + '.log'))
    logger = logging.getLogger(logger_name)
    dataloader = get_test_loader(config.testsets_path, batch_size=config.batch_size, shuffle=False)
    
    # Model config depending on dataset
    model_config = dict(
        model_path=config.model_path,
        num_channels=128,
        num_res_blocks=1,
        attention_resolutions="16",
    ) if config.model_name == 'ffhq_10m' \
        else dict(
        model_path=config.model_path,
        num_channels=256,
        num_res_blocks=2,
        attention_resolutions="8,16,32",
    )
    
    args = utils_model.create_argparser(model_config).parse_args([])
    unet, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()))
    unet.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    unet.eval()
    unet = unet.to(device)

    shutil.copyfile(config.opt, os.path.join(config.save_path, os.path.basename('config.yaml')))

    operator = get_operator(config.operator_name, config=config, logger=logger, device=device)
    operator.model = operator.model.to(device)
    ns = NoiseSchedule(config, logger, device)

    cond_method = get_conditioning_method(name=config.conditioning_method)
    cond_method = cond_method.conditioning
    
    p_sample_loop(config, ns, unet, diffusion, operator, cond_method, dataloader, device, logger)


if __name__ == '__main__':
    main()