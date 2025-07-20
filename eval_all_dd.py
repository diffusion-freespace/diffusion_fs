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
from metrics import *

class Config:
    def __init__(self, config):
        self.config = config
        for k, v in self.config.items():
            self.__setattr__(k, v)

config = "configs/carla.yaml"
situation_classes_root = "situation_classes/"

with open(config, "r") as f:
    config = Config(yaml.safe_load(f))
    
dataset_class = {"nuscenes" : NuScenesDataset, "carla" : FSCacheDataset}

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
situations = ["0", "1", "2", "3", "4"]

for situation in situations:
    all_eval = [{'eval_noise' : False, 'ckpt_path' : "pretrained_ckpts/carla_base.ckpt", 'name' : "base"},
                {'eval_noise' : False, 'ckpt_path' : "pretrained_ckpts/carla_cls.ckpt", 'name' : "cls"},
                {'eval_noise' : True, 'ckpt_path' : "pretrained_ckpts/carla_base.ckpt", 'name' : "noise"},
                {'eval_noise' : False, 'ckpt_path' : "pretrained_ckpts/carla_base.ckpt", 'name' : "obs"},
                {'eval_noise' : False, 'ckpt_path' : "pretrained_ckpts/carla_cls.ckpt", 'name' : "obs_cls"}]

    base_path = "results"
    base_path = os.path.join(base_path, f"sit_{situation}")
    os.makedirs(base_path, exist_ok=True)

    for eval_config in all_eval:
        EVAL_NOISE = eval_config['eval_noise']
        ckpt_path = eval_config['ckpt_path']
        name = eval_config['name']
        situation_path = os.path.join(base_path, name + ".txt")

        if(EVAL_NOISE):
            templatedataset = FSCacheDataset("datasets_all/noise_template.yaml", is_train=False)
            temploader = torch.utils.data.dataloader.DataLoader(templatedataset, batch_size=1, num_workers=16, shuffle=False)

        with open(config.dataset_config, 'r') as f:
            config_ = yaml.safe_load(f)
            config_['data_dir_test'] = f"{situation_classes_root}/{situation}/"
        
        with open(config.dataset_config, "w") as f:
            yaml.dump(config_, f)

        val_dataset = dataset_class[config.dataset_type](config.dataset_config, is_train=False)
        valloader = torch.utils.data.dataloader.DataLoader(val_dataset, batch_size=1, num_workers=4, shuffle=False, worker_init_fn=seed_worker)
        # model = LITFSModel.load_from_checkpoint("../pretrained_ckpts/last_carla_contour_resnet18_base-v1.ckpt", strict=False)
        model = LITFSModel.load_from_checkpoint(ckpt_path, strict=False)
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

        # os.makedirs(situation_path,exist_ok=True)
        # csv_path = os.path.join(situation_path, "tangents.csv")

        # with open(csv_path, "w", newline="") as f:
        #     writer = csv.writer(f)
        #     writer.writerow(["idx"] + [f"tangent_{i}" for i in range(12)])

        count = 0
        all_final_metrics = {}
        seed = 2025

        all_dd_vals = []
        L.seed_everything(seed)
        final_metrics = {"val_iou": 0, "count": 0, "val_obs_overlap" : 0, "val_offroad_overlap" : 0}

        with torch.no_grad():
            for idx, batch in enumerate(tqdm(valloader)):
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
                
                generator = torch.Generator("cuda")
                
                if(EVAL_NOISE):
                    template_noise_t_ = template_noise_t.clone().repeat(b, 1, 1, 1)
                    pred_contours = model.model.infer_from_template_noise(img, template_noise_t_, begin_t=time)
                else:
                    if("obs" in name):
                        pred_contours = model.model.obstacle_guidance(img, obstacles.cuda(), cmd=cmd)
                    else:
                        pred_contours = model.model.infer(img, cmd=cmd) # 6b, 1, 50, 2
                
                pred_segs_metrics = pred_contours.clone().cpu().numpy()
            
                pred_segs_metrics[..., 1] = ((pred_segs_metrics[..., 1] + 1) / 2) * image_size[0]
                pred_segs_metrics[..., 0] = ((pred_segs_metrics[..., 0] + 1) / 2) * image_size[1]
                    
                if isinstance(pred_segs_metrics, torch.Tensor):
                    pred_segs_metrics = pred_segs_metrics.cpu().numpy().astype(np.int32)
                else:
                    pred_segs_metrics = pred_segs_metrics.astype(np.int32)
                    
                masks = []
                for i in range(len(img)):
                    mask = np.zeros((image_size[0], image_size[1]))
                    cv2.drawContours(mask, [pred_segs_metrics[i].squeeze()], -1, 1, -1)
                    masks.append(mask)
                masks = np.array(masks)
                metrics = compute_metrics(torch.from_numpy(masks), seg, obstacles)

                for k, v in final_metrics.items():
                    if(k == "count"):
                        continue
                    final_metrics[k] = (final_metrics[k] * final_metrics['count'] + metrics[k] * (len(masks))) / (final_metrics['count'] + len(masks))
                final_metrics["count"] += len(masks)
                    
                # vis_imgs = model.visualize(pred_contours, img)
                tangent_values = [idx]
                
                for j, img_ in enumerate(img):
                    contour = pred_contours[j].squeeze()
                    contour[..., 1] = ((contour[..., 1] + 1) / 2) * image_size[0]
                    contour[..., 0] = ((contour[..., 0] + 1) / 2) * image_size[1]
                    centerline = compute_centerline_centroid(contour, bin_size=20)

                    if len(centerline) < 3:
                        tangent_values.append(np.nan)
                    else:
                        theta_start = compute_tangent_angle(centerline)
                        tangent_values.append(theta_start)
                
                all_dd_vals.append(tangent_values[1:])
                
                # with open(csv_path, "a", newline="") as f:
                #     writer = csv.writer(f)
                #     writer.writerow(tangent_values)

        all_dd_vals = np.array(all_dd_vals)
        extent = np.nanmax(all_dd_vals, -1) - np.nanmin(all_dd_vals, -1)
        mean_x = np.nanmean(all_dd_vals, -1)
        std_x = np.nanstd(all_dd_vals, -1)

        mean = np.nanmean(mean_x)
        std = np.nanmean(std_x)
        mean_ext = np.nanmean(extent)
        print(f"Mean: {mean}, Std: {std}, Mean Extent: {mean_ext}")

        for k, v in final_metrics.items():
            print(k, v)
            
        with open(situation_path, "w") as f:
            f.write(f"Mean: {mean}, Std: {std}, Mean Extent: {mean_ext}\n")
            for k, v in final_metrics.items():
                f.write(f"{k}: {v}\n")