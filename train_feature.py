
import os
from unittest.mock import patch
from tqdm.auto import tqdm
from opt import config_parser



import json, random
from renderer import *
from utils import *
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF
import datetime

from dataLoader import dataset_dict
import sys



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

renderer = OctreeRender_trilinear_fast


class SimpleSampler:
    def __init__(self, total, batch):
        self.total = total
        self.batch = batch
        self.curr = total
        self.ids = None

    def nextids(self):
        self.curr+=self.batch
        if self.curr + self.batch > self.total:
            self.ids = torch.LongTensor(np.random.permutation(self.total))
            self.curr = 0
        return self.ids[self.curr:self.curr+self.batch]

def InfiniteSampler(n):
    # i = 0
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0

class InfiniteSamplerWrapper(torch.utils.data.sampler.Sampler):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31

@torch.no_grad()
def render_test(args):
    # init dataset
    dataset = dataset_dict[args.dataset_name]
    test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train, is_stack=True)
    white_bg = test_dataset.white_bg
    ndc_ray = args.ndc_ray

    if not os.path.exists(args.ckpt):
        print('the ckpt path does not exists!!')
        return

    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.change_to_feature_mod(args.n_lamb_sh ,device)
    tensorf.load(ckpt)
    tensorf.eval()
    tensorf.rayMarch_weight_thres = args.rm_weight_mask_thre

    logfolder = os.path.dirname(args.ckpt)


    if args.render_train:
        os.makedirs(f'{logfolder}/{args.expname}/imgs_train_all', exist_ok=True)
        train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True)
        evaluation_feature(train_dataset,tensorf, args, renderer, args.chunk_size, f'{logfolder}/{args.expname}/imgs_train_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)

    if args.render_test:
        os.makedirs(f'{logfolder}/{args.expname}/imgs_test_all', exist_ok=True)
        evaluation_feature(test_dataset,tensorf, args, renderer, args.chunk_size, f'{logfolder}/{args.expname}/imgs_test_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)

    if args.render_path:
        c2ws = test_dataset.render_path
        os.makedirs(f'{logfolder}/{args.expname}/imgs_path_all', exist_ok=True)
        evaluation_feature_path(test_dataset,tensorf, c2ws, renderer, args.chunk_size, f'{logfolder}/{args.expname}/imgs_path_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)

def reconstruction(args):

    # init dataset
    dataset = dataset_dict[args.dataset_name]
    train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True)
    white_bg = train_dataset.white_bg
    near_far = train_dataset.near_far
    h_rays, w_rays = train_dataset.img_wh[1], train_dataset.img_wh[0]
    ndc_ray = args.ndc_ray

    patch_size = args.patch_size

    
    if args.add_timestamp:
        logfolder = f'{args.basedir}/{args.expname}{datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")}'
    else:
        logfolder = f'{args.basedir}/{args.expname}'
    

    # init log file
    os.makedirs(logfolder, exist_ok=True)
    summary_writer = SummaryWriter(logfolder)



    # init parameters
    # tensorVM, renderer = init_parameters(args, train_dataset.scene_bbox.to(device), reso_list[0])
    aabb = train_dataset.scene_bbox.to(device)
    # TODO: need to update reso_cur
    reso_cur = N_to_reso(args.N_voxel_init, aabb)
    nSamples = min(args.nSamples, cal_n_samples(reso_cur,args.step_ratio))


    assert args.ckpt is not None, 'Have to be pre-trained to get density fielded!'

    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device':device})
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.load(ckpt)
   
    tensorf.change_to_feature_mod(args.n_lamb_sh ,device)
    tensorf.rayMarch_weight_thres = args.rm_weight_mask_thre

    train_dataset.prepare_feature_data(tensorf.encoder)

    grad_vars = tensorf.get_optparam_groups_feature_mod(args.lr_init, args.lr_basis)
    if args.lr_decay_iters > 0:
        lr_factor = args.lr_decay_target_ratio**(1/args.lr_decay_iters)
    else:
        args.lr_decay_iters = args.n_iters
        lr_factor = args.lr_decay_target_ratio**(1/args.n_iters)

    print("lr decay", args.lr_decay_target_ratio, args.lr_decay_iters)
    
    optimizer = torch.optim.Adam(grad_vars, betas=(0.9,0.99))


    torch.cuda.empty_cache()
    PSNRs = []

    allrays, allfeatures = train_dataset.all_rays, train_dataset.all_features
    allrays_stack, allrgbs_stack = train_dataset.all_rays_stack, train_dataset.all_rgbs_stack
    if not args.ndc_ray:
        allrays, allfeatures = tensorf.filtering_rays(allrays, allfeatures, bbox_only=True)
    trainingSampler = SimpleSampler(allrays.shape[0], args.batch_size)
    frameSampler = iter(InfiniteSamplerWrapper(allrays_stack.size(0))) # every next(sampler) returns a frame index


    TV_weight_feature = args.TV_weight_feature
    tvreg = TVLoss()
    print(f"initial TV_weight_feature: {TV_weight_feature}")


    pbar = tqdm(range(args.n_iters), miniters=args.progress_refresh_rate, file=sys.stdout)
    for iteration in pbar:

        feature_loss, pixel_loss = 0., 0.
        if iteration%2==0:
            ray_idx = trainingSampler.nextids()
            rays_train, features_train = allrays[ray_idx], allfeatures[ray_idx].to(device)

            feature_map, _ = renderer(rays_train, tensorf, chunk=args.chunk_size, N_samples=nSamples, white_bg = white_bg, 
                                ndc_ray=ndc_ray, render_feature=True, device=device, is_train=True)

            feature_loss = torch.mean((feature_map - features_train) ** 2)
        else:
            frame_idx = next(frameSampler)
            start_h = np.random.randint(0, h_rays-patch_size+1)
            start_w = np.random.randint(0, w_rays-patch_size+1)
            if white_bg:
                # move random sampled patches into center
                mid_h, mid_w = (h_rays-patch_size+1)/2, (w_rays-patch_size+1)/2
                if mid_h-start_h>=1:
                    start_h += np.random.randint(0, mid_h-start_h)
                elif mid_h-start_h<=-1:
                    start_h += np.random.randint(mid_h-start_h, 0)
                if mid_w-start_w>=1:
                    start_w += np.random.randint(0, mid_w-start_w)
                elif mid_w-start_w<=-1:
                    start_w += np.random.randint(mid_w-start_w, 0)

            rays_train = allrays_stack[frame_idx, start_h:start_h+patch_size, 
                                                    start_w:start_w+patch_size, :].reshape(-1, 6).to(device)
            # [patch*patch, 6]
            
            rgbs_train = allrgbs_stack[frame_idx, start_h:(start_h+patch_size), 
                                                  start_w:(start_w+patch_size), :].to(device)
            # [patch, patch, 3]

            feature_map, _ = renderer(rays_train, tensorf, chunk=args.chunk_size, N_samples=nSamples, white_bg=white_bg, 
                                ndc_ray=ndc_ray, render_feature=True, device=device, is_train=True)

            feature_map = feature_map.reshape(patch_size, patch_size, 256)[None,...].permute(0,3,1,2)
            recon_rgb = tensorf.decoder(feature_map)

            rgbs_train = rgbs_train[None,...].permute(0,3,1,2)
            img_enc = tensorf.encoder(normalize_vgg(rgbs_train))
            recon_rgb_enc = tensorf.encoder(recon_rgb)
            
            feature_loss =(F.mse_loss(recon_rgb_enc.relu4_1, img_enc.relu4_1) +
                           F.mse_loss(recon_rgb_enc.relu3_1, img_enc.relu3_1)) / 10

            recon_rgb = denormalize_vgg(recon_rgb)

            pixel_loss = torch.mean((recon_rgb - rgbs_train) ** 2)

        total_loss = pixel_loss + feature_loss

        # loss
        # NOTE: Calculate feature TV loss rather than appearence TV loss
        if TV_weight_feature>0:
            TV_weight_feature *= lr_factor
            loss_tv = tensorf.TV_loss_feature(tvreg)*TV_weight_feature
            total_loss = total_loss + loss_tv
            summary_writer.add_scalar('train/reg_tv_feature', loss_tv.detach().item(), global_step=iteration)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if pixel_loss == 0:
            feature_loss = feature_loss.detach().item()
            PSNRs.append(-10.0 * np.log(feature_loss) / np.log(10.0))
            summary_writer.add_scalar('train/PSNR_feature', PSNRs[-1], global_step=iteration)
            summary_writer.add_scalar('train/mse_feature', feature_loss, global_step=iteration)
        else:
            pixel_loss = pixel_loss.detach().item()
            PSNRs.append(-10.0 * np.log(pixel_loss) / np.log(10.0))
            summary_writer.add_scalar('train/PSNR_pixel', PSNRs[-1], global_step=iteration)
            summary_writer.add_scalar('train/mse_pixel', pixel_loss, global_step=iteration)
            summary_writer.add_scalar('train/mse_recon_feature', feature_loss.detach().item(), global_step=iteration)


        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * lr_factor

        # Print the current values of the losses.
        if iteration % args.progress_refresh_rate == 0:
            pbar.set_description(
                f'Iteration {iteration:05d}:'
                + f' psnr = {float(np.mean(PSNRs)):.2f}'
            )
            PSNRs = []

        if iteration % (args.progress_refresh_rate*20) == 1:
            summary_writer.add_image('output', make_grid([rgbs_train.squeeze(), 
                                                        recon_rgb.clamp(0, 1).squeeze()],  
                                                        nrow=2, padding=0, normalize=False),
                                                        global_step=iteration)
        

    tensorf.save(f'{logfolder}/{args.expname}.th')

if __name__ == '__main__':

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20211202)
    np.random.seed(20211202)

    args = config_parser()
    print(args)

    if args.render_only:
        render_test(args)
    else:
        reconstruction(args)

