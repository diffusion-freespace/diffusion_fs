import torch
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from diffusers import DDPMScheduler
from metrics import compute_metrics
from transformers import  AutoImageProcessor, \
    EfficientNetModel, SwinModel, \
         ViTModel, MobileNetV2Model, AutoModel

class Scheduler():
    def __init__(self, config):
        self.config = config
        self.T = config.num_train_timesteps
        self.betas = torch.linspace(self.config.beta_start, self.config.beta_end, steps=self.T)
        self.alphas = 1 - self.betas
        self.num_train_timesteps = self.T
        self.timesteps = torch.arange(self.T).flip(0)
        self.alpha_bars = torch.cumprod(self.alphas, 0)
    
    def add_noise(self, sample, noise, timesteps):
        mean = torch.sqrt(self.alpha_bars.to(noise)[timesteps]).to(noise)[:, None, None, None] * sample
        std = torch.sqrt(1 - self.alpha_bars.to(noise)[timesteps]).to(noise)
        return noise*std[:, None, None, None] + mean
    
    def step(self, pred_noise, t, sample):
        noise_add = torch.randn_like(sample)
        if(t != 0):
            denoised = 1/torch.sqrt(self.alphas.to(pred_noise)[t]) * (sample - self.betas.to(pred_noise)[t]/torch.sqrt(1 - self.alpha_bars.to(pred_noise)[t]) * pred_noise) + noise_add * torch.sqrt(self.betas.to(pred_noise)[t]).to(sample)
        else:
            denoised = 1/torch.sqrt(self.alphas.to(pred_noise)[t]) * (sample - self.betas.to(pred_noise)[t]/torch.sqrt(1 - self.alpha_bars.to(pred_noise)[t]) * pred_noise)
        return denoised
    
class GeneralEncoder(nn.Module):
    def __init__(self, backbone = 'resnet18', pretrained = True, num_images=1, init_ch=3):
        super(GeneralEncoder, self).__init__()
        print("inside general encoder class")
        self.backbone = backbone
        # breakpoint()
        if 'resnet' in backbone:
            self.img_preprocessor = None
            self.encoder = ResNetEncoder(backbone=backbone,
                                         pretrained=pretrained,
                                         num_images = num_images,
                                         init_ch=init_ch)
            self.encoder_dims = 512
        elif backbone == 'efficientnet':
            self.img_preprocessor = AutoImageProcessor.from_pretrained("google/efficientnet-b0")
            self.encoder = EfficientNetModel.from_pretrained("google/efficientnet-b0") 
            self.encoder_dims = 1280
        elif backbone == 'swinmodel':
            self.img_preprocessor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
            self.encoder = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
            self.encoder_dims = 768
        elif backbone == 'vit':
            self.img_preprocessor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
            self.encoder = ViTModel.from_pretrained("google/vit-base-patch16-224")
            self.encoder_dims = 768
        elif backbone == 'mobilenet':
            self.encoder_dims = 1280
            self.img_preprocessor = AutoImageProcessor.from_pretrained("google/mobilenet_v2_1.0_224")
            self.encoder = MobileNetV2Model.from_pretrained("google/mobilenet_v2_1.0_224")
        elif backbone == 'dino':
            self.encoder_dims = 768
            self.img_preprocessor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
            self.encoder = AutoModel.from_pretrained('facebook/dinov2-base')
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
    
    def forward(self, x, return_all=False):
        if 'resnet' in self.backbone:
            return self.encoder(x, return_all)
        
        device = x.device
        x = self.img_preprocessor(x, return_tensors = 'pt')
        pixel_values = x['pixel_values'].to(device)
        enc_output = self.encoder(pixel_values=pixel_values)
        outputs = enc_output.last_hidden_state
        
        if self.backbone == 'vit':
            reshaped_tensor = outputs.permute(0, 2, 1)[:, :, 1:].reshape(-1, 768, 14, 14)
            return reshaped_tensor
        
        if self.backbone == 'swinmodel':
            reshaped_tensor = outputs.permute(0, 2, 1).reshape(-1, 768, 7, 7)
            return reshaped_tensor
        
        if self.backbone == 'dino':
            reshaped_tensor = outputs.permute(0, 2, 1)[:, :, 1:].reshape(-1, 768, 16, 16)
            return reshaped_tensor
        
        return outputs
            
        
class ResNetEncoder(nn.Module):
    def __init__(self, backbone='resnet18', pretrained=True, num_images=1, init_ch=3):
        super(ResNetEncoder, self).__init__()
        
        # Load the pre-trained ResNet model
        if backbone == 'resnet18':
            self.model = models.resnet18(pretrained=pretrained)
        elif backbone == 'resnet34':
            self.model = models.resnet34(pretrained=pretrained)
        elif backbone == 'resnet50':
            self.model = models.resnet50(pretrained=pretrained)
        elif backbone == 'resnet101':
            self.model = models.resnet101(pretrained=pretrained)
        elif backbone == 'resnet152':
            self.model = models.resnet152(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        if(num_images > 1):
            self.model.conv1 = nn.Conv2d(init_ch*num_images, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False).to(self.model.conv1.weight.device)
        self.layer0 = nn.Sequential(self.model.conv1, self.model.bn1, self.model.relu, self.model.maxpool)
        self.layer1 = self.model.layer1
        self.layer2 = self.model.layer2
        self.layer3 = self.model.layer3
        self.layer4 = self.model.layer4
        
    def preprocess(self, x):
        x = x/255
        return x

    def forward(self, x, return_all=False):
        x = x / 255 # self.preprocess(x)
        outputs = {}
        x0 = self.layer0(x)  # First downsample: output after conv1, bn1, relu, and maxpool
        x1 = self.layer1(x0)  # Second downsample: layer1
        x2 = self.layer2(x1)  # Third downsample: layer2
        x3 = self.layer3(x2)  # Fourth downsample: layer3
        x4 = self.layer4(x3)  # Final downsample: layer4

        outputs[0], outputs[1], outputs[2], outputs[3], outputs[4] = x0, x1, x2, x3, x4

        if(return_all):
            return outputs
        return outputs[4] #downstream, only 4 is being used

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.q = nn.Linear(config.query_dim, config.query_dim)
        self.k = nn.Linear(config.query_dim, config.query_dim)
        self.v = nn.Linear(config.query_dim, config.query_dim)
        assert config.attention_emb_dim % config.mha_heads == 0, "mha_heads must be divisible by attention_emb_dim"
        self.mha = nn.MultiheadAttention(config.attention_emb_dim, config.mha_heads, batch_first=True)
        self.out_linear = nn.Linear(config.attention_emb_dim, config.query_dim)
    
    def forward(self, q, k, v, return_attn_maps=False):
        out, attn_maps = self.mha(self.q(q), self.k(k), self.v(v), need_weights=return_attn_maps)
        out = self.out_linear(out)
        if(return_attn_maps):
            return out, attn_maps
        return out

class TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.norm1 = nn.LayerNorm(config.query_dim)
        self.norm2 = nn.LayerNorm(config.query_dim)
        # ca and sa block
        if(config.conditioning == "cls"):
            self.pos = nn.Parameter(torch.randn(1, config.num_contour_points + 2, config.query_dim))
        else:
            self.pos = nn.Parameter(torch.randn(1, config.num_contour_points + 1, config.query_dim))
        self.sa = Attention(config)
        self.ca = Attention(config)
        self.ff1 = nn.Linear(config.query_dim, 2*config.query_dim)
        self.ff2 = nn.Linear(2*config.query_dim, config.query_dim)
        
    def forward(self, queries, img_feats, return_attn_maps=False):
        queries = self.norm1(queries) + self.pos
        queries_new = self.sa(queries, queries, queries)
        queries = queries_new + queries
        
        queries = self.norm2(queries)
        if(return_attn_maps):
            queries_new, attn_maps = self.ca(queries, img_feats, img_feats, return_attn_maps=return_attn_maps)
        else:
            queries_new = self.ca(queries, img_feats, img_feats, return_attn_maps=return_attn_maps)
        queries = queries_new + queries
        queries = self.ff2(F.relu(self.ff1(queries))) + queries
        if(return_attn_maps):
            return queries, attn_maps
        return queries
    
def fourier_embedding(x, D):
    # freqs = torch.tensor([2**i for i in range(D // 2)], dtype=torch.float32).to(x.device)[None]
    freqs = torch.tensor([i+1 for i in range(D // 2)], dtype=torch.float32).to(x.device)[None]
    emb_sin = torch.sin(freqs * x)
    emb_cos = torch.cos(freqs * x)
    embedding = torch.cat([emb_sin, emb_cos], dim=-1)
    
    return embedding


class DiffusionFS(nn.Module):
    def __init__(self, config):
        super(DiffusionFS, self).__init__()
        self.config = config
        if(self.config.dataset_type == "nuscenes"):
            self.config.img_size = tuple([256, 512])
        elif(self.config.dataset_type == "carla"):
            self.config.img_size = tuple([288, 768])
        else:
            raise ValueError(f"Unsupported dataset type: {self.config.dataset_type}")
        
        print(f"Using backbone :{config.backbone}")
        self.backbone = GeneralEncoder(config.backbone)
        self.img_proj = nn.Conv2d(self.backbone.encoder_dims, config.query_dim//2, kernel_size=1)

        self.pe_lin = nn.Linear(2, config.query_dim//2)
        self.pre_predict = nn.Linear(config.query_dim, 2)
        
        self.out = nn.Linear(self.config.query_dim + self.config.query_dim, 2)
        self.tr = nn.ModuleList([TransformerLayer(config) for _ in range(config.num_tr_layers)])
        
        self.scheduler = DDPMScheduler(num_train_timesteps=config.num_train_timesteps,
                                       beta_start=config.beta_start,
                                       beta_end=config.beta_end,
                                       beta_schedule=config.beta_schedule)
        
        # self.scheduler = Scheduler(config)
        
        if(self.config.conditioning == "cls"):
            self.num_classes = 6
            self.class_proj = torch.nn.Linear(self.num_classes, self.config.query_dim) 

    def forward(self, seg, img, t, img_feat=None, class_labels=None):
        b, _, n, _ = seg.shape

        if(self.config.conditioning == "cls"):
            class_one_hot = F.one_hot(class_labels, num_classes=self.num_classes).float()  # (b, num_classes)
            class_embedding = self.class_proj(class_one_hot).unsqueeze(1)  # (b, 1, c)
        
        if(img_feat is None):
            img_feat = self.img_proj(self.backbone(img)) # (b, c//2, h//32, w//32)

        pos_seg = self.pe_lin(seg) # (b, 1, n, c//2)
        seg_features_from_image = F.grid_sample(img_feat, seg, padding_mode="border") # (b, c//2, 1, n)
        seg_tokens = torch.cat([pos_seg[:, 0], seg_features_from_image[:, :, 0].permute(0, 2, 1)], -1) # (b, n, c)
        
        new_tokens = seg_tokens
        init_pred = self.pre_predict(new_tokens)

        t_emb = fourier_embedding(t.float()[:, None], self.config.query_dim)[:, None]
        all_tokens = torch.cat([new_tokens, t_emb], 1)
        
        if(self.config.conditioning == "cls"):
            all_tokens = torch.cat([all_tokens, class_embedding], 1)

        for tr in self.tr:
            all_tokens = tr(all_tokens, all_tokens, return_attn_maps=False)

        denoised_seg = self.out(torch.cat([all_tokens[:, :n], t_emb.repeat(1, new_tokens.shape[1], 1)], -1)).reshape(b, 1, self.config.num_contour_points, 2)
        denoised_seg = denoised_seg + init_pred[:, None]

        return denoised_seg
    
    def compute_loss(self, batch, is_train=True):
        output = {}
        data = batch
        seg, img = data['contour'], data['img']
        valid = data['valid']
        batch_mask = valid
        seg = seg[batch_mask]
        img = img[batch_mask]
        if(self.config.conditioning == "cls"):
            assert 'cmd' in data, "cmd must be present in the batch for class conditioning"
            cmd = data['cmd'][batch_mask]
        else:
            cmd = None
        
        if(len(img) == 0):
            output['loss'] = 0
            return output
        
        b = seg.shape[0]
        timestep = torch.randint(0, self.scheduler.num_train_timesteps - 1, size=(b,)).to(img.device)
        noise = torch.randn_like(seg)
        noisy_sample = self.scheduler.add_noise(seg, noise=noise, timesteps=timestep)
        pred_noise = self(noisy_sample, img, timestep, img_feat=None, class_labels=cmd)

        loss = F.mse_loss(pred_noise, noise)
        output['loss'] = loss
        return output

    
    def validate(self, batch):
        output = self.compute_loss(batch, is_train=False)
        return output
    
    def infer(self, img, generator=None, cmd=None, all_timesteps=False, avg=1):
        img_feat = None
        b, _, h, w = img.shape
        
        for attr_name, attr_value in self.scheduler.__dict__.items():
            if isinstance(attr_value, torch.Tensor):
                self.scheduler.__dict__[attr_name] = attr_value.cuda()
        
        if(all_timesteps):
            all_timestep_maps = []

        seg_map = torch.randn(b*avg, 1, self.config.num_contour_points, 2).to(img)
        img_feat = self.img_proj(self.backbone(img)).repeat(avg, 1, 1, 1)
        
        if(self.config.conditioning == "cls"):
            assert cmd is not None, "cmd must be provided for class conditioning"
            cmd = cmd.repeat(avg)
        
        for time in self.scheduler.timesteps:
            t = torch.tensor([time]).to(img).long().repeat(b*avg)
            pred_noise = self(seg_map, img, t, img_feat=img_feat, class_labels=cmd)
            t = torch.tensor([time]).to(img).long()
            seg_map = self.scheduler.step(pred_noise, t, seg_map, generator=generator).prev_sample
            if(all_timesteps):
                all_timestep_maps.append(seg_map)
        
        seg_map = torch.stack(torch.chunk(seg_map, avg)).permute(1, 0, 2, 3, 4).mean(1)
        if(all_timesteps):
            return seg_map, torch.stack(all_timestep_maps)

        return seg_map
    
    def obstacle_guidance(self, img, obstacle_mask, cmd=None, generator=None, avg=1):
        img_feat = None
        b, _, h, w = img.shape
        
        obstacle_mask = obstacle_mask[:, :2].sum(1)
        obstacle_mask[obstacle_mask > 1] = 1
        
        for attr_name, attr_value in self.scheduler.__dict__.items():
            if isinstance(attr_value, torch.Tensor):
                self.scheduler.__dict__[attr_name] = attr_value.cuda()
            
        seg_map = torch.randn(b*avg, 1, self.config.num_contour_points, 2).to(img)
        img_feat = self.img_proj(self.backbone(img)).repeat(avg, 1, 1, 1)
        
        for time in self.scheduler.timesteps:
            t = torch.tensor([time]).to(img).long().repeat(b*avg)
            pred_noise = self(seg_map, img, t, img_feat=img_feat, class_labels=cmd)
            # points lying inside the obstacle_mask
            curr_int_points = (((seg_map + 1) / 2) * 1)[:, 0]
            curr_outside_points_mask_ = ((curr_int_points < 0) | (curr_int_points >= 1))
            curr_outside_points_mask = torch.zeros(b, self.config.num_contour_points).to(torch.bool)

            curr_outside_points_mask[curr_outside_points_mask_.sum(-1) >= 1] = 1
            curr_outside_points_mask = curr_outside_points_mask.to(torch.bool)
            
            curr_int_points[curr_outside_points_mask_] = 0
            curr_points_x = ((curr_int_points).reshape(b, -1, 2)[:, :, 0] * self.config.img_size[1]).to(torch.int32)
            curr_points_y = ((curr_int_points).reshape(b, -1, 2)[:, :, 1] * self.config.img_size[0]).to(torch.int32)
            batch_indices = torch.arange(b).to(torch.int32)[None].T.repeat(1, self.config.num_contour_points)
            vals_inside_obstacle = obstacle_mask[batch_indices, curr_points_y, curr_points_x]
            vals_inside_obstacle[curr_outside_points_mask] = 0 # Points lying outside the image won't be under a mask
            vals_inside_obstacle = vals_inside_obstacle[:, None, :, None].repeat(1, 1, 1, 2)
            
            other_means = seg_map[:, 0].mean(-2) # (b, 2)
            diff_means = other_means[:, None, None] - seg_map # (b, 1, 50, 2)
            
            pred_noise[vals_inside_obstacle == 0] = pred_noise[vals_inside_obstacle == 0]
            pred_noise[vals_inside_obstacle == 1] =  -4 * diff_means[vals_inside_obstacle == 1] + pred_noise[vals_inside_obstacle == 1]
            
            # pred_noise = pred_noise * (vals_inside_obstacle)
            
            t = torch.tensor([time]).to(img).long()
            # print(pred_noise.shape)
            seg_map = self.scheduler.step(pred_noise, t, seg_map, generator=generator).prev_sample
        
        seg_map = torch.stack(torch.chunk(seg_map, avg)).permute(1, 0, 2, 3, 4).mean(1)

        return seg_map
    
    def shift_points_while_inferring(self, img, cmd=None):
        img_feat = None
        b, _, h, w = img.shape
        
        for attr_name, attr_value in self.scheduler.__dict__.items():
            if isinstance(attr_value, torch.Tensor):
                self.scheduler.__dict__[attr_name] = attr_value.cuda()
            
        seg_map = torch.randn(b, 1, self.config.num_contour_points, 2).to(img)
        img_feat = self.img_proj(self.backbone(img))
        
        if(self.config.conditioning == "cls"):
            assert cmd is not None, "cmd must be provided for class conditioning"
        
        for time in self.scheduler.timesteps:
            t = torch.tensor([time]).to(img).long().repeat(b)
            pred_noise = self(seg_map, img, t, img_feat=img_feat, class_labels=cmd)
            t = torch.tensor([time]).to(img).long()
            seg_map = self.scheduler.step(pred_noise, t, seg_map).prev_sample

        return seg_map
    
    def infer_from_template_noise(self, img, template_noise, begin_t, cmd=None, return_all_timesteps=False):
        img_feat = None
        b, _, h, w = img.shape
        
        for attr_name, attr_value in self.scheduler.__dict__.items():
            if isinstance(attr_value, torch.Tensor):
                self.scheduler.__dict__[attr_name] = attr_value.cuda()
            
        seg_map = template_noise.to(img)
        img_feat = self.img_proj(self.backbone(img))
        
        if(return_all_timesteps):
            all_timestep_maps = []
            
        for time in self.scheduler.timesteps:
            if(time > begin_t):
                continue
            t = torch.tensor([time]).to(img).long().repeat(b)
            pred_noise = self(seg_map, img, t, img_feat=img_feat, class_labels=cmd)
            t = torch.tensor([time]).to(img).long()
            seg_map = self.scheduler.step(pred_noise, t, seg_map).prev_sample
            if(return_all_timesteps):
                all_timestep_maps.append(seg_map)

        if(return_all_timesteps):
            return seg_map, torch.stack(all_timestep_maps)
        
        return seg_map
    
    def add_noise(self, seg, timestep):
        noise = torch.randn_like(seg)
        noisy_sample = self.scheduler.add_noise(seg, noise=noise, timesteps=timestep)
        
        return noisy_sample
    
    def get_feat_maps(self, img):
        img_feat = self.backbone(img)
        return img_feat
            

class LITFSModel(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        self.model = DiffusionFS(config)
        self.model = torch.compile(self.model)
        
    def training_step(self, batch, idx):
        output = self.model.compute_loss(batch)
        train_loss = output['loss']
        
        self.log("train_loss", train_loss, sync_dist=True, prog_bar=True)
        return train_loss
    
    def validation_step(self, batch, idx):
        metrics = self.eval_batch(batch, idx, split="val")
        for k, v in metrics.items():
            self.log(k, v, sync_dist=True, prog_bar=True)
                
    def test_step(self, batch, idx):
        metrics = self.eval_batch(batch, idx, split="test")
        for k, v in metrics.items():
            self.log(k, v, sync_dist=True, prog_bar=True)
    
    def eval_batch(self, batch, idx, split="val", avg=1):
        valid = batch['valid']
        seg, img, obstacles = batch['mask'], batch['img'], batch['obstacles']
        seg = seg[valid]
        img = img[valid]
        obstacles = obstacles[valid]
        if(self.config.conditioning == "cls"):
            cmd = batch['cmd'][valid]
        else:
            cmd = None
        
        if(len(img) == 0):
            return {}
        
        pred_contours = self.model.infer(img, cmd=cmd, avg=avg)
        # pred_contours = self.model.obstacle_guidance(img, obstacle_mask=obstacles, cmd=cmd)
        
        if(idx == 0):
            vis_imgs = self.visualize(pred_contours, img)
            for i, img_ in enumerate(vis_imgs):
                cv2.imwrite(f"vis2/{i}.png", img_)
                
        pred_contours = pred_contours.cpu().numpy()
        
        pred_contours[...,  0] = ((pred_contours[...,  0] + 1)/2 * self.config.img_size[1])
        pred_contours[..., 1] = ((pred_contours[...,  1] + 1)/2 * self.config.img_size[0])
        pred_contours = pred_contours.astype(np.int32)
        masks = []
        for i in range(len(img)):
            mask = np.zeros((img[i].shape[1], img[i].shape[2]))
            cv2.drawContours(mask, [pred_contours[i].squeeze()], -1, 1, -1)
            masks.append(mask)
        masks = np.array(masks)
        metrics = compute_metrics(masks, seg.cpu().numpy(), obstacles.cpu().numpy(), split=split)
        
        return metrics
    
    def visualize(self, pred_seg, img):
        imgs = img.permute(0, 2, 3, 1).cpu().numpy()
        pred_seg = pred_seg.permute(0, 2, 3, 1).cpu().numpy()
        all_vis = []
        for i, img in enumerate(imgs):
            img = img.astype(np.uint8)
            img = np.ascontiguousarray(img)
            pred_seg_ = np.squeeze(pred_seg[i], axis=-1)
            pred_seg_[:, 0] = ((pred_seg_[:, 0] + 1)/2 * self.config.img_size[1])
            pred_seg_[:, 1] = ((pred_seg_[:, 1] + 1)/2 * self.config.img_size[0])
            pred_seg_ = pred_seg_.astype(np.int32)
            cv2.drawContours(img, [pred_seg_], -1, (255, 0, 0), -1)
            all_vis.append(img)
            
        return all_vis
    
    def visualize_templates_in_one(self, pred_seg, img):
        imgs = img.transpose(0, 2, 3, 1)
        b, h, w, _ = imgs.shape
        pred_seg = pred_seg.transpose(0, 2, 3, 1)

        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

        all_vis = []
        alpha = 0.5

        for i, img in enumerate(imgs):
            img = img.astype(np.uint8)
            img = np.ascontiguousarray(img)
            overlay = img.copy()
            
        for j in range(pred_seg.shape[0]):
            pred_seg_ = np.squeeze(pred_seg[j], axis=-1)
            contour = np.zeros_like(pred_seg_)
            contour[:, 0] = (pred_seg_[:, 0] + 1) / 2 * w
            contour[:, 1] = (pred_seg_[:, 1] + 1) / 2 * h
            contour = contour.astype(np.int32)

            cv2.drawContours(overlay, [contour], -1, colors[j], thickness=-1)

        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
        all_vis.append(img)
        
        return all_vis, overlay
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), self.config.learning_rate)
        return optimizer
    