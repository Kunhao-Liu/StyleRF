import torch,os,imageio,sys
from tqdm.auto import tqdm
from dataLoader.ray_utils import get_rays
from models.tensoRF import raw2alpha, TensorVMSplit, AlphaGridMask
from utils import *
from dataLoader.ray_utils import ndc_rays_blender, denormalize_vgg, normalize_vgg


def OctreeRender_trilinear_fast(rays, tensorf, chunk=4096, N_samples=-1, ndc_ray=False, white_bg=True, is_train=False, 
                                render_feature=False, style_img=None, device='cuda'):

    rgbs, alphas, depth_maps, weights, uncertainties = [], [], [], [], []
    features, accs = [], []
    s_mean_std_mat = None
    if style_img is not None:
        with torch.no_grad():
            style_feature = tensorf.encoder(normalize_vgg(style_img))
        s_mean_std_mat = tensorf.stylizer.get_style_mean_std_matrix(style_feature.relu3_1.flatten(2))

    N_rays_all = rays.shape[0]
    for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
        rays_chunk = rays[chunk_idx * chunk:(chunk_idx + 1) * chunk].to(device)
        
        if render_feature:
            feature_map, acc_map = tensorf.render_feature_map(rays_chunk, s_mean_std_mat=s_mean_std_mat, is_train=is_train, ndc_ray=ndc_ray, N_samples=N_samples)
            features.append(feature_map)
            accs.append(acc_map)
        else:
            rgb_map, depth_map = tensorf(rays_chunk, is_train=is_train, white_bg=white_bg, ndc_ray=ndc_ray, N_samples=N_samples)
            rgbs.append(rgb_map)
            depth_maps.append(depth_map)
    
    if render_feature:
        if style_img is not None:
            return torch.cat(features), torch.cat(accs), style_feature
        return torch.cat(features), torch.cat(accs)

    return torch.cat(rgbs), None, torch.cat(depth_maps), None, None 

def OctreeRender_trilinear_fast_depth(rays, tensorf, chunk=4096, N_samples=-1, ndc_ray=False, is_train=False, device='cuda'):

    depth_maps = []
    N_rays_all = rays.shape[0]
    for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
        rays_chunk = rays[chunk_idx * chunk:(chunk_idx + 1) * chunk].to(device)
        depth_map = tensorf.render_depth_map(rays_chunk, is_train=is_train, ndc_ray=ndc_ray, N_samples=N_samples)
        depth_maps.append(depth_map)
    
    return torch.cat(depth_maps)


@torch.no_grad()
def evaluation_feature(test_dataset, tensorf, args, renderer, chunk_size=2048, savePath=None, N_vis=10, prtx='', N_samples=-1,
               white_bg=False, ndc_ray=False, compute_extra_metrics=False, style_img=None, device='cuda'):
    '''
    To see if the decoded feature map is similar to gt rgb map
    '''
    PSNRs, rgb_maps, vis_feature_maps = [], [], []
    ssims,l_alex,l_vgg=[],[],[]
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath + '/feature', exist_ok=True)
    W, H = test_dataset.img_wh

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    img_eval_interval = 1 if N_vis < 0 else max(test_dataset.all_rays_stack.shape[0] // N_vis,1)
    idxs = list(range(0, test_dataset.all_rays_stack.shape[0], img_eval_interval))
    for idx, samples in tqdm(enumerate(test_dataset.all_rays_stack[0::img_eval_interval]), file=sys.stdout):        

        rays = samples.view(-1,samples.shape[-1])

        if style_img is None:
            feature_map, _ = renderer(rays, tensorf, chunk=chunk_size, N_samples=N_samples, ndc_ray=ndc_ray, 
                                        white_bg = white_bg, render_feature=True, device=device)
        else:
            feature_map, _, _ = renderer(rays, tensorf, chunk=chunk_size, N_samples=N_samples, ndc_ray=ndc_ray, 
                                white_bg = white_bg, render_feature=True, style_img=style_img, device=device)
                            
        feature_map = feature_map.reshape(H, W, 256)[None,...].permute(0,3,1,2)

        recon_rgb = denormalize_vgg(tensorf.decoder(feature_map))
        recon_rgb = recon_rgb.permute(0,2,3,1).clamp(0,1)

        vis_feature_map = torch.sigmoid(feature_map[:, [1,2,3], :, :].permute(0,2,3,1))
        
        if test_dataset.white_bg:
            mask = test_dataset.all_masks[idx:idx+1].to(device)
            recon_rgb = mask*recon_rgb + (1.-mask)
            vis_feature_map = mask*vis_feature_map + (1.-mask)

        recon_rgb = recon_rgb.reshape(H, W, 3).cpu()
        vis_feature_map = vis_feature_map.squeeze().cpu()

        if len(test_dataset.all_rgbs_stack):
            gt_rgb = test_dataset.all_rgbs_stack[idxs[idx]].view(H, W, 3)
            loss = torch.mean((recon_rgb - gt_rgb) ** 2)
            PSNRs.append(-10.0 * np.log(loss.item()) / np.log(10.0))

            if compute_extra_metrics:
                ssim = rgb_ssim(recon_rgb, gt_rgb, 1)
                l_a = rgb_lpips(gt_rgb.numpy(), recon_rgb.numpy(), 'alex', tensorf.device)
                l_v = rgb_lpips(gt_rgb.numpy(), recon_rgb.numpy(), 'vgg', tensorf.device)
                ssims.append(ssim)
                l_alex.append(l_a)
                l_vgg.append(l_v)

        recon_rgb = (recon_rgb.numpy() * 255).astype('uint8')
        vis_feature_map = (vis_feature_map.numpy() * 255).astype('uint8')
        gt_rgb = (gt_rgb.numpy() * 255).astype('uint8')

        if savePath is not None:
            # rgb_map = np.concatenate((recon_rgb, gt_rgb), axis=1)
            rgb_maps.append(recon_rgb)
            vis_feature_maps.append(vis_feature_map)
            imageio.imwrite(f'{savePath}/{prtx}{idx:03d}.png', recon_rgb)
            imageio.imwrite(f'{savePath}/feature/feature_{idx:03d}.png', vis_feature_map)

    imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(rgb_maps), fps=30, quality=8)
    imageio.mimwrite(f'{savePath}/feature/feature_video.mp4', np.stack(vis_feature_maps), fps=30, quality=8)

    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr, ssim, l_a, l_v]))
        else:
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr]))


    return PSNRs

@torch.no_grad()
def evaluation_feature_path(test_dataset, tensorf, c2ws, renderer, chunk_size=2048, savePath=None, N_vis=5, prtx='', N_samples=-1,
                    white_bg=False, ndc_ray=False, compute_extra_metrics=False, style_img=None, device='cuda'):
    PSNRs, rgb_maps, vis_feature_maps, depth_maps = [], [], [], []
    ssims,l_alex,l_vgg=[],[],[]
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath + '/feature', exist_ok=True)
    W, H = test_dataset.img_wh

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    for idx, c2w in tqdm(enumerate(c2ws)):

        c2w = torch.FloatTensor(c2w)
        rays_o, rays_d = get_rays(test_dataset.directions, c2w)  # both (h*w, 3)
        if ndc_ray:
            rays_o, rays_d = ndc_rays_blender(H, W, test_dataset.focal[0], 1.0, rays_o, rays_d)
        rays = torch.cat([rays_o, rays_d], 1).reshape(H, W, 6).permute(2,0,1)  # (6,H,W)
        rays = rays.permute(1,2,0).reshape(-1,6) # (H * W, 6)

        if style_img is None:
            feature_map, _ = renderer(rays, tensorf, chunk=chunk_size, N_samples=N_samples, ndc_ray=ndc_ray, 
                                        white_bg = white_bg, render_feature=True, device=device)
        else:
            feature_map, _, _ = renderer(rays, tensorf, chunk=chunk_size, N_samples=N_samples, ndc_ray=ndc_ray, 
                                white_bg = white_bg, render_feature=True, style_img=style_img, device=device)
        
        feature_map = feature_map.reshape(H, W, 256)[None,...].permute(0,3,1,2)

        recon_rgb = denormalize_vgg(tensorf.decoder(feature_map))
        recon_rgb = recon_rgb.permute(0,2,3,1).clamp(0,1)
        recon_rgb = recon_rgb.reshape(H, W, 3).cpu()
        recon_rgb = (recon_rgb.numpy() * 255).astype('uint8')
        rgb_maps.append(recon_rgb)

        vis_feature_map = torch.sigmoid(feature_map[:, [1,2,3], :, :].permute(0,2,3,1))
        vis_feature_map = vis_feature_map.squeeze().cpu()
        vis_feature_map = (vis_feature_map.numpy() * 255).astype('uint8')
        vis_feature_maps.append(vis_feature_map)
        
        if savePath is not None:
            imageio.imwrite(f'{savePath}/{prtx}{idx:03d}.png', recon_rgb)
            imageio.imwrite(f'{savePath}/feature/feature_{idx:03d}.png', vis_feature_map)

    imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(rgb_maps), fps=30, quality=8)
    imageio.mimwrite(f'{savePath}/feature/feature_video.mp4', np.stack(vis_feature_maps), fps=30, quality=8)

    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr, ssim, l_a, l_v]))
        else:
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr]))


    return PSNRs

@torch.no_grad()
def evaluation(test_dataset,tensorf, args, renderer, savePath=None, N_vis=5, prtx='', N_samples=-1,
               white_bg=False, ndc_ray=False, compute_extra_metrics=True, device='cuda'):
    PSNRs, rgb_maps, depth_maps = [], [], []
    ssims,l_alex,l_vgg=[],[],[]
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath+"/rgbd", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    img_eval_interval = 1 if N_vis < 0 else max(test_dataset.all_rays_stack.shape[0] // N_vis,1)
    idxs = list(range(0, test_dataset.all_rays_stack.shape[0], img_eval_interval))
    for idx, samples in tqdm(enumerate(test_dataset.all_rays_stack[0::img_eval_interval]), file=sys.stdout):

        W, H = test_dataset.img_wh
        rays = samples.view(-1,samples.shape[-1])

        rgb_map, _, depth_map, _, _ = renderer(rays, tensorf, chunk=4096, N_samples=N_samples,
                                        ndc_ray=ndc_ray, white_bg = white_bg, device=device)
        rgb_map = rgb_map.clamp(0.0, 1.0)

        rgb_map, depth_map = rgb_map.reshape(H, W, 3).cpu(), depth_map.reshape(H, W).cpu()

        depth_map, _ = visualize_depth_numpy(depth_map.numpy(),near_far)
        if len(test_dataset.all_rgbs_stack):
            gt_rgb = test_dataset.all_rgbs_stack[idxs[idx]].view(H, W, 3)
            loss = torch.mean((rgb_map - gt_rgb) ** 2)
            PSNRs.append(-10.0 * np.log(loss.item()) / np.log(10.0))

            if compute_extra_metrics:
                ssim = rgb_ssim(rgb_map, gt_rgb, 1)
                l_a = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'alex', tensorf.device)
                l_v = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'vgg', tensorf.device)
                ssims.append(ssim)
                l_alex.append(l_a)
                l_vgg.append(l_v)

        rgb_map = (rgb_map.numpy() * 255).astype('uint8')
        # rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map)
        if savePath is not None:
            imageio.imwrite(f'{savePath}/{prtx}{idx:03d}.png', rgb_map)
            # rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
            imageio.imwrite(f'{savePath}/rgbd/{prtx}{idx:03d}.png', depth_map)

    imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(rgb_maps), fps=30, quality=10)
    imageio.mimwrite(f'{savePath}/{prtx}depthvideo.mp4', np.stack(depth_maps), fps=30, quality=10)

    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr, ssim, l_a, l_v]))
        else:
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr]))


    return PSNRs

@torch.no_grad()
def evaluation_path(test_dataset,tensorf, c2ws, renderer, savePath=None, N_vis=5, prtx='', N_samples=-1,
                    white_bg=False, ndc_ray=False, compute_extra_metrics=True, device='cuda'):
    PSNRs, rgb_maps, depth_maps = [], [], []
    ssims,l_alex,l_vgg=[],[],[]
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath+"/rgbd", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    for idx, c2w in tqdm(enumerate(c2ws)):

        W, H = test_dataset.img_wh

        c2w = torch.FloatTensor(c2w)
        rays_o, rays_d = get_rays(test_dataset.directions, c2w)  # both (h*w, 3)
        if ndc_ray:
            rays_o, rays_d = ndc_rays_blender(H, W, test_dataset.focal[0], 1.0, rays_o, rays_d)
        rays = torch.cat([rays_o, rays_d], 1)  # (h*w, 6)

        rgb_map, _, depth_map, _, _ = renderer(rays, tensorf, chunk=8192, N_samples=N_samples,
                                        ndc_ray=ndc_ray, white_bg = white_bg, device=device)
        rgb_map = rgb_map.clamp(0.0, 1.0)

        rgb_map, depth_map = rgb_map.reshape(H, W, 3).cpu(), depth_map.reshape(H, W).cpu()

        depth_map, _ = visualize_depth_numpy(depth_map.numpy(),near_far)

        rgb_map = (rgb_map.numpy() * 255).astype('uint8')
        # rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map)
        if savePath is not None:
            imageio.imwrite(f'{savePath}/{prtx}{idx:03d}.png', rgb_map)
            rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
            imageio.imwrite(f'{savePath}/rgbd/{prtx}{idx:03d}.png', rgb_map)

    imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(rgb_maps), fps=30, quality=8)
    imageio.mimwrite(f'{savePath}/{prtx}depthvideo.mp4', np.stack(depth_maps), fps=30, quality=8)

    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr, ssim, l_a, l_v]))
        else:
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr]))


    return PSNRs

