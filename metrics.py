import torch
import numpy as np

def compute_centerline_centroid(contour, bin_size=5):
    contour = contour.cpu().numpy()
    contour = contour[np.argsort(contour[:, 1])]

    y_min, y_max = np.min(contour[:, 1]), np.max(contour[:, 1])
    y_bins = np.arange(y_min, y_max + bin_size, bin_size)

    centerline = []
    for i in range(len(y_bins) - 1):
        y_lower, y_upper = y_bins[i], y_bins[i + 1]

        bin_points = contour[(contour[:, 1] >= y_lower) & (contour[:, 1] < y_upper)]
        if len(bin_points) == 0:
            continue
        centroid_x = np.mean(bin_points[:, 0])
        centroid_y = np.mean(bin_points[:, 1])

        centerline.append([centroid_x, centroid_y])

    return np.array(centerline, dtype=np.float32)

def compute_curvature(contour, eps=1e-8):
    contour = contour.astype(np.float32)
    dx = np.gradient(contour[:, 0])
    dy = np.gradient(contour[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    curvature = np.abs(dx * ddy -dy * ddx) / (np.power(dx**2 + dy**2, 1.5)+ eps)
    
    return curvature

def compute_tangent_diff(contour):
    contour = contour.astype(np.float32)
    start_vec = contour[1] - contour[0]
    end_vec = contour[-1] - contour[-2]
    start_angle = np.arctan2(start_vec[1], start_vec[0])
    end_angle = np.arctan2(end_vec[1], end_vec[0])
    delta_angle = np.arctan2(np.sin(end_angle - start_angle), np.cos(end_angle - start_angle))
    
    return delta_angle

def compute_tangent_angle(centerline):
    start = centerline[0]
    end = centerline[-1]

    theta_start = np.arctan2(end[1] - start[1], end[0] - start[0]) * (180 / np.pi)

    return theta_start

def compute_iou(pred_masks, gt_masks):
    intersection = pred_masks * gt_masks
    union = pred_masks + gt_masks
    union[union > 0] = 1
    iou = (intersection.sum(1).sum(1) + 1e-6) / (union.sum(1).sum(1) + 1e-6)
    return iou
    
def compute_overlap(pred_masks, gt_masks):
    intersection = pred_masks * gt_masks
    overlap = (intersection.sum(1).sum(1) + 1e-6) / (pred_masks.sum(1).sum(1) + 1e-6)
    filter = pred_masks.sum(1).sum(1) > 0
    overlap[~filter] = 0
    return overlap

def compute_metrics(pred_masks, gt_masks, obstacles, split="val"):
    metrics = {}
    iou = compute_iou(pred_masks, gt_masks).mean()

    obstacle_ped_vcls = obstacles[:, 0] + obstacles[:, 1]
    obstacle_ped_vcls[obstacle_ped_vcls > 1] = 1
    obstacle_overlap = compute_overlap(pred_masks, obstacle_ped_vcls).mean()
    
    road_mask = obstacles[:, 2]
    offroad_overlap = compute_overlap(pred_masks, 1 - road_mask).mean()
    
    metrics[f'{split}_iou'] = iou
    metrics[f'{split}_obs_overlap'] = obstacle_overlap
    metrics[f'{split}_offroad_overlap'] = offroad_overlap
    
    return metrics