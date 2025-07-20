import torch
import numpy as np
import cv2
import lightning as L
import os
from tqdm import tqdm
from datasets_all.nusc_dataset import NuScenesDataset
import yaml
from fs_model import LITFSModel
import random
from datasets_all.fs_cache_dataset import FSCacheDataset
from metrics import compute_metrics

def gaussian_smooth_contour(contour, sigma=-1):
    smoothed_x = gaussian_filter1d(contour[:, 0], sigma=sigma, mode='nearest')
    smoothed_y = gaussian_filter1d(contour[:, 1], sigma=sigma, mode='nearest')

    smooth_contour = np.vstack((smoothed_x, smoothed_y)).T
    return smooth_contour.reshape(-1, 1, 2).astype(np.int32)

class Config:
    def __init__(self, config):
        self.config = config
        for k, v in self.config.items():
            self.__setattr__(k, v)

config = "configs/carla.yaml"
ckpt_path = "pretrained_ckpts/carla_base.ckpt"
EVAL_NOISE = False
OBS_GUIDANCE = False

# config = "configs/nuscenes.yaml"
# ckpt_path = "pretrained_ckpts/nuscenes_base.ckpt"
# EVAL_NOISE = False
# OBS_GUIDANCE = False

output_folder = "visualizations"
os.makedirs(output_folder, exist_ok=True)

with open(config, "r") as f:
    config = Config(yaml.safe_load(f))
    
dataset_class = {"nuscenes" : NuScenesDataset, "carla" : FSCacheDataset}

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    

if(EVAL_NOISE):
    templatedataset = FSCacheDataset("datasets_all/noise_template.yaml", is_train=False)
    temploader = torch.utils.data.dataloader.DataLoader(templatedataset, batch_size=1, num_workers=16, shuffle=False)

val_dataset = dataset_class[config.dataset_type](config.dataset_config, is_train=False)
valloader = torch.utils.data.dataloader.DataLoader(val_dataset, batch_size=1, num_workers=4, shuffle=False, worker_init_fn=seed_worker)
model = LITFSModel.load_from_checkpoint(ckpt_path, strict=False)
# model = LITFSModel.load_from_checkpoint("pretrained_ckpts/nuscenes_base.ckpt", strict=False)
model.cuda()
model.eval()
image_size = model.config.img_size

if(EVAL_NOISE):
    template_dir = "templates"
    os.makedirs(template_dir, exist_ok=True)
    for i in range(6):
        os.makedirs(f"{template_dir}/cmd_{i}", exist_ok=True)
        
    time = 10
    with torch.no_grad():
        count = 0
        cmd_segments = {i: [] for i in range(6)}

        for batch in tqdm(temploader):
            seg, img, cmd = batch['contour'], batch['img'].cuda(), batch['cmd']
            i = cmd[0].item()
            cmd_segments[i].extend(seg)
            count += 1
            if all(len(cmd_segments[i]) >= 5 for i in range(6)):
                break

        template_noise=[]
        normal_template_noies=[]
        for i in range(6):
            if len(cmd_segments[i]) > 0:
                seg = torch.cat(cmd_segments[i])
                timestep = torch.ones(len(seg)).to(torch.long) * time
                noise = model.model.add_noise(seg, timestep)
                noise = noise.mean(0)
                template_noise.append(noise) # 1, 50, 2
                
                noise = noise.clone().cpu().numpy()
                noise[:, 0] = (noise[:, 0] + 1) / 2 * image_size[1]
                noise[:, 1] = (noise[:, 1] + 1) / 2 * image_size[0]
                noise = noise.astype(np.int32)
                img = np.zeros((image_size[0], image_size[1]))

                for point in noise:
                    cv2.circle(img, tuple(point), 1, 1, 1)

                cv2.imwrite(f"{template_dir}/cmd_{i}/template.png", img * 255)

    template_noise_t = torch.stack(template_noise, dim=0).squeeze(1)
    template_noise_t = template_noise_t.unsqueeze(1)

count = 0
all_final_metrics = {}
seeds = [2025]

for seed in seeds:
    L.seed_everything(seed)
    final_metrics = {"val_iou": 0, "count": 0, "val_obs_overlap" : 0, "val_offroad_overlap" : 0}
    
    with torch.no_grad():
        for batch in tqdm(valloader):
            data = batch
            seg, img = data['mask'], data['img'].cuda()
            b = img.shape[0]
            obstacles = data['obstacles']
            valid = data['valid']
            seg = seg[valid].repeat(6, 1, 1)
            img = img[valid].repeat(6, 1, 1, 1)
            obstacles = obstacles[valid].repeat(6, 1, 1, 1)

            if(model.config.conditioning == "cls"):
                cmd = torch.arange(6).long().cuda().repeat(b)
            else:
                cmd = None
            
            if(len(img) == 0):
                continue

            if(EVAL_NOISE):
                template_noise_t_ = template_noise_t.clone().repeat(b, 1, 1, 1)
                pred_contours = model.model.infer_from_template_noise(img, template_noise_t_, begin_t=time, cmd=cmd)
            else:
                if(OBS_GUIDANCE):
                    pred_contours = model.model.obstacle_guidance(img, obstacles.cuda(), cmd=cmd)
                else:
                    pred_contours = model.model.infer(img, cmd=cmd) # 6b, 1, 50, 2
            
            for i in range(b):
                pred_contour = pred_contours[i*6:(i+1)*6].cpu().numpy()
                
                all_vis, overlay = model.visualize_templates_in_one(pred_contour, img[i*6:(i+1)*6].cpu().numpy())
                cv2.imwrite(f"{output_folder}/vis_{count}.png", overlay)
                count += 1
            