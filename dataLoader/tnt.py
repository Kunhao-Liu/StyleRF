import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import os
from PIL import Image
from torchvision import transforms as T
import numpy as np
import glob
import imageio
from .ray_utils import *
########################################################################################################################
# camera coordinate system: x-->right, y-->down, z-->scene (opencv/colmap convention)
# poses is camera-to-world
########################################################################################################################
def find_files(dir, exts):
    if os.path.isdir(dir):
        # types should be ['*.png', '*.jpg']
        files_grabbed = []
        for ext in exts:
            files_grabbed.extend(glob.glob(os.path.join(dir, ext)))
        if len(files_grabbed) > 0:
            files_grabbed = sorted(files_grabbed)
        return files_grabbed
    else:
        return []
    
def parse_txt(filename):
    assert os.path.isfile(filename)
    nums = open(filename).read().split()
    return np.array([float(x) for x in nums]).reshape([4, 4]).astype(np.float32)

def get_rays_single_image(H, W, intrinsics, c2w):
    '''
    :param H: image height
    :param W: image width
    :param intrinsics: 4 by 4 intrinsic matrix
    :param c2w: 4 by 4 camera to world extrinsic matrix
    :return:
    '''
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    u = u.reshape(-1).astype(dtype=np.float32) + 0.5    # add half pixel
    v = v.reshape(-1).astype(dtype=np.float32) + 0.5
    pixels = np.stack((u, v, np.ones_like(u)), axis=0)  # (3, H*W)

    rays_d = np.dot(np.linalg.inv(intrinsics[:3, :3]), pixels)
    rays_d = np.dot(c2w[:3, :3], rays_d)  # (3, H*W)
    rays_d = rays_d.transpose((1, 0))  # (H*W, 3)

    rays_o = c2w[:3, 3].reshape((1, 3))
    rays_o = np.tile(rays_o, (rays_d.shape[0], 1))  # (H*W, 3)

    depth = np.linalg.inv(c2w)[2, 3]
    depth = depth * np.ones((rays_o.shape[0],), dtype=np.float32)  # (H*W,)

    return rays_o, rays_d, depth

class tnt(Dataset):
    def __init__(self, datadir, split='train', downsample=4, is_stack=False, skip=1):

        self.root_dir = datadir
        self.split = split
        self.is_stack = is_stack
        self.downsample = downsample

        self.white_bg = False
        self.near_far = [0.0, 256]
        self.scene_bbox = torch.tensor([[-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]])

        self.read_data(skip=skip)

    
    def read_data(self, skip=1):
        split_dir = os.path.join(self.root_dir, self.split)
        # camera parameters files
        intrinsics_files = find_files('{}/intrinsics'.format(split_dir), exts=['*.txt'])
        pose_files = find_files('{}/pose'.format(split_dir), exts=['*.txt'])
        print('raw intrinsics_files: {}'.format(len(intrinsics_files)))
        print('raw pose_files: {}'.format(len(pose_files)))

        intrinsics_files = intrinsics_files[::skip]
        pose_files = pose_files[::skip]
        cam_cnt = len(pose_files)
        
        # image files
        img_files = find_files('{}/rgb'.format(split_dir), exts=['*.png', '*.jpg'])
        print('raw img_files: {}'.format(len(img_files)))
        img_files = img_files[::skip]
        assert(len(img_files) == cam_cnt)

        train_im = imageio.imread(img_files[0])
        H, W = train_im.shape[:2]
        self.img_wh = np.array([int(W // self.downsample), int(H // self.downsample)])

        self.all_rays = []
        self.all_rgbs = []
        for i in tqdm(range(cam_cnt)):
            # read rgbs
            img = self.read_rgbs(img_files[i]) # (h*w, 3)
            self.all_rgbs += [img]

            # read rays
            intrinsics = parse_txt(intrinsics_files[i])
            intrinsics[:2, :3] /= self.downsample

            pose = parse_txt(pose_files[i])
            c2w_mat = pose

            rays_o, rays_d, _ = get_rays_single_image(self.img_wh[1], self.img_wh[0], intrinsics, c2w_mat) # (h*w, 3) (h*w, 3)
            rays_o = torch.from_numpy(rays_o)
            rays_d = torch.from_numpy(rays_d)
            self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)

        all_rays = self.all_rays
        all_rgbs = self.all_rgbs

        self.all_rays = torch.cat(self.all_rays, 0) # (len(self.meta['frames])*h*w,6)
        self.all_rgbs = torch.cat(self.all_rgbs, 0) # (len(self.meta['frames])*h*w,3)

        if self.is_stack:
            self.all_rays_stack = torch.stack(all_rays, 0).reshape(-1,*self.img_wh[::-1], 6)  # (len(self.meta['frames]),h,w,6)
            avg_pool = torch.nn.AvgPool2d(4, ceil_mode=True)
            self.ds_all_rays_stack = avg_pool(self.all_rays_stack.permute(0,3,1,2)).permute(0,2,3,1) # (len(self.meta['frames]),h/4,w/4,6)
            self.all_rgbs_stack = torch.stack(all_rgbs, 0).reshape(-1,*self.img_wh[::-1], 3)  # (len(self.meta['frames]),h,w,3)


    @torch.no_grad()
    def prepare_feature_data(self, encoder, chunk=8):
        '''
        Prepare feature maps as training data.
        '''
        assert self.is_stack, 'Dataset should contain original stacked taining data!'
        print('====> prepare_feature_data ...')

        frames_num, h, w, _ = self.all_rgbs_stack.size()
        features = []

        for chunk_idx in tqdm(range(frames_num // chunk + int(frames_num % chunk > 0))):
            rgbs_chunk = self.all_rgbs_stack[chunk_idx*chunk : (chunk_idx+1)*chunk].cuda()
            features_chunk = encoder(normalize_vgg(rgbs_chunk.permute(0,3,1,2))).relu3_1
            # resize to the size of rgb map so that rays can match
            features_chunk = T.functional.resize(features_chunk, size=(h,w), 
                                                 interpolation=T.InterpolationMode.BILINEAR)
            features.append(features_chunk.detach().cpu().requires_grad_(False))

        self.all_features_stack = torch.cat(features).permute(0,2,3,1) # (len(self.meta['frames]),h,w,256)
        self.all_features = self.all_features_stack.reshape(-1, 256)
        print('prepare_feature_data Done!')
        

    def read_rgbs(self, img_file):
        img = Image.open(img_file).convert('RGB')
        if self.downsample != 1.0:
            img = img.resize(self.img_wh, Image.LANCZOS)
        img = T.ToTensor()(img)
        img = img.view(3, -1).permute(1, 0)  # (h*w, 3) RGB

        return img