
import os
from tqdm.auto import tqdm
from opt import config_parser
from PIL import Image, ImageFile
from pathlib import Path
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF

from renderer import *
from utils import *
from torch.utils.tensorboard import SummaryWriter
import datetime

from dataLoader import dataset_dict
from dataLoader.styleLoader import getDataLoader

from models.styleModules import cal_mse_content_loss, cal_adain_style_loss

import sys



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

renderer = OctreeRender_trilinear_fast
depth_renderer = OctreeRender_trilinear_fast_depth


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
    ndc_ray = args.ndc_ray

    if not os.path.exists(args.ckpt):
        print('the ckpt path does not exists!!')
        return

    assert args.style_img is not None, 'Must specify a style image!'

    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.change_to_feature_mod(args.n_lamb_sh, device)
    tensorf.change_to_style_mod(device)
    tensorf.load(ckpt)
    tensorf.eval()
    tensorf.rayMarch_weight_thres = args.rm_weight_mask_thre

    logfolder = os.path.dirname(args.ckpt)

    trans = T.Compose([T.Resize(size=(256,256)), T.ToTensor()])
    style_img = trans(Image.open(args.style_img)).cuda()[None, ...]
    style_name = Path(args.style_img).stem

    if args.render_train:
        train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True)
        os.makedirs(f'{logfolder}/{args.expname}/imgs_train_all/{style_name}', exist_ok=True)
        evaluation_feature(train_dataset,tensorf, args, renderer, args.chunk_size, f'{logfolder}/{args.expname}/imgs_train_all/{style_name}',
                                N_vis=-1, N_samples=-1, white_bg = train_dataset.white_bg, ndc_ray=ndc_ray, style_img=style_img, device=device)
    
    if args.render_test:
        test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train, is_stack=True)
        os.makedirs(f'{logfolder}/{args.expname}/imgs_test_all/{style_name}', exist_ok=True)
        evaluation_feature(test_dataset,tensorf, args, renderer, args.chunk_size, f'{logfolder}/{args.expname}/imgs_test_all/{style_name}',
                                N_vis=-1, N_samples=-1, white_bg = test_dataset.white_bg, ndc_ray=ndc_ray, style_img=style_img, device=device)

    if args.render_path:
        test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train, is_stack=True)
        c2ws = test_dataset.render_path
        os.makedirs(f'{logfolder}/{args.expname}/imgs_path_all/{style_name}', exist_ok=True)
        evaluation_feature_path(test_dataset, tensorf, c2ws, renderer, args.chunk_size, f'{logfolder}/{args.expname}/imgs_path_all/{style_name}',
                N_vis=-1, N_samples=-1, white_bg = test_dataset.white_bg, ndc_ray=ndc_ray, style_img=style_img, device=device)

def reconstruction(args):

    # init dataset
    dataset = dataset_dict[args.dataset_name]
    train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True)
    white_bg = train_dataset.white_bg
    near_far = train_dataset.near_far
    h_rays, w_rays = train_dataset.img_wh[1], train_dataset.img_wh[0]
    ndc_ray = args.ndc_ray

    patch_size = args.patch_size # ground truth image patch size when training

    Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
    ImageFile.LOAD_TRUNCATED_IMAGES = True # Disable OSError: image file is truncated
    style_loader = getDataLoader(args.wikiartdir, batch_size=1, sampler=InfiniteSamplerWrapper, 
                    image_side_length=256, num_workers=2)
    style_iter = iter(style_loader)
    
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
    tensorf.change_to_feature_mod(args.n_lamb_sh, device)
    tensorf.load(ckpt)
    tensorf.change_to_style_mod(device)
    tensorf.rayMarch_weight_thres = args.rm_weight_mask_thre

    tvreg = TVLoss()

    grad_vars = tensorf.get_optparam_groups_style_mod(args.lr_basis, args.lr_finetune)
    if args.lr_decay_iters > 0:
        lr_factor = args.lr_decay_target_ratio**(1/args.lr_decay_iters)
    else:
        args.lr_decay_iters = args.n_iters
        lr_factor = args.lr_decay_target_ratio**(1/args.n_iters)

    print("lr decay", args.lr_decay_target_ratio, args.lr_decay_iters)
    
    tensorf.train()
    optimizer = torch.optim.Adam(grad_vars, betas=(0.9,0.99))

    torch.cuda.empty_cache()

    allrays_stack, allrgbs_stack = train_dataset.all_rays_stack, train_dataset.all_rgbs_stack
    frameSampler = iter(InfiniteSamplerWrapper(allrays_stack.size(0))) # every next(sampler) returns a frame index

    pbar = tqdm(range(args.n_iters), miniters=args.progress_refresh_rate, file=sys.stdout)
    for iteration in pbar:

        # get style_img, this style_img has NOT been normalized according to the pretrained VGGmodel
        style_img = next(style_iter)[0].to(device)

        # randomly sample patch_size*patch_size patch from given frame
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

        rays_train = allrays_stack[frame_idx, start_h:start_h+patch_size, start_w:start_w+patch_size, :]\
                            .reshape(-1, 6).to(device)
        # [patch*patch, 6]
        
        rgbs_train = allrgbs_stack[frame_idx, start_h:(start_h+patch_size), 
                                            start_w:(start_w+patch_size), :].to(device)
        # [patch, patch, 3]

        feature_map, acc_map, style_feature = renderer(rays_train, tensorf, chunk=args.chunk_size, N_samples=nSamples, white_bg = white_bg, 
                                ndc_ray=ndc_ray, render_feature=True, style_img=style_img, device=device, is_train=True)

        feature_map = feature_map.reshape(patch_size, patch_size, 256)[None,...].permute(0,3,1,2)
        rgb_map = tensorf.decoder(feature_map)

        # feature_map is trained with normalized rgb maps, so here we don't normalize the rgb map again.
        rgbs_train = normalize_vgg(rgbs_train[None,...].permute(0,3,1,2))
        
        out_image_feature = tensorf.encoder(rgb_map)
        content_feature = tensorf.encoder(rgbs_train)

        if white_bg:
            mask = acc_map.reshape(patch_size, patch_size, 1)[None,...].permute(0,3,1,2)
            if not (mask>0.5).any(): continue
            
            # content loss
            _mask = F.interpolate(mask, size=content_feature.relu4_1.size()[-2:], mode='bilinear').ge(1e-5)
            content_loss = cal_mse_content_loss(torch.masked_select(content_feature.relu4_1, _mask), 
                                                torch.masked_select(out_image_feature.relu4_1, _mask))
            # style loss
            style_loss = 0.
            for style_feature, image_feature in zip(style_feature, out_image_feature):
                _mask = F.interpolate(mask, size=image_feature.size()[-2:], mode='bilinear').ge(1e-5)
                C = image_feature.size()[1]
                masked_img_feature = torch.masked_select(image_feature, _mask).reshape(1,C,-1)
                style_loss += cal_adain_style_loss(style_feature, masked_img_feature)

            content_loss *= args.content_weight
            style_loss *= args.style_weight
        else:
            # content loss
            content_loss = cal_mse_content_loss(content_feature.relu4_1, out_image_feature.relu4_1)
            # style loss
            style_loss = 0.
            for style_feature, image_feature in zip(style_feature, out_image_feature):
                style_loss += cal_adain_style_loss(style_feature, image_feature)

            content_loss *= args.content_weight
            style_loss *= args.style_weight

        feature_tv_loss = tvreg(feature_map) * args.featuremap_tv_weight
        image_tv_loss = tvreg(denormalize_vgg(rgb_map)) * args.image_tv_weight

        total_loss = content_loss + style_loss + feature_tv_loss + image_tv_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * lr_factor
      
        # Print the current values of the losses.
        if iteration%args.progress_refresh_rate==0:
            summary_writer.add_scalar('train/content_loss', content_loss, global_step=iteration)
            summary_writer.add_scalar('train/style_loss', style_loss, global_step=iteration)
            summary_writer.add_scalar('train/feature_tv_loss', feature_tv_loss, global_step=iteration)
            summary_writer.add_scalar('train/image_tv_loss', image_tv_loss, global_step=iteration)
            pbar.set_description(
                f'Iteration {iteration:05d}:'
                + f' content_loss = {content_loss.item():.2f}'
                + f' style_loss = {style_loss.item():.2f}'
            )
       
        if iteration % (args.progress_refresh_rate*20) == 0:
            summary_writer.add_image('output', make_grid([denormalize_vgg(rgbs_train).squeeze(), \
                                                denormalize_vgg(rgb_map).clamp(0, 1).squeeze(), \
                                                TF.resize(style_img, (patch_size,patch_size)).squeeze()],  
                                                nrow=3, padding=0, normalize=False),
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

