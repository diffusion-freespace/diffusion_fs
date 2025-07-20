import numpy as np
import cv2
import torch
import sys
from tqdm import tqdm
import os
from basic_dataset import BasicDataset
from skimage.morphology import skeletonize

def extract_roi_above_ego(combined_rgb, ego_x=160, ego_y=280, roi_width=200, roi_height=150):
    H, W, _ = combined_rgb.shape
    x_min = max(0, ego_x - roi_width // 2)
    x_max = min(W, ego_x + roi_width // 2)
    y_min = max(0, ego_y - roi_height)
    y_max = ego_y
    
    roi = combined_rgb[y_min:y_max, x_min:x_max, :].copy()  
    mask = roi != 0
    roi[mask] = 75 
    return roi, (x_min, y_min, x_max, y_max)
    
def extract_lane_segments_from_roi(roi):
    gray = cv2.cvtColor((roi * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    binary_image = (gray > 0).astype(np.uint8)
    skeleton = skeletonize(binary_image).astype(np.uint8) * 255
    edges = cv2.Canny(skeleton, 50, 150)
    
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=5, minLineLength=5, maxLineGap=5)
    lane_segments = [(x1, y1, x2, y2) for line in lines for x1, y1, x2, y2 in line] if lines is not None else []
    return lane_segments

def compute_equation(x1, y1, x2, y2):
    A = y2 - y1
    B = -(x2 - x1)
    C = -(A * x1 + B * y1)
    return A, B, C


def convert_to_cartesian(lines, H):
    cartesian_lines = []
    for x1, y1, x2, y2 in lines:
        cartesian_lines.append((x1, H - y1, x2, H - y2))
   
    return cartesian_lines
        
def point_to_line_distance(px, py, x1, y1, x2, y2):
    if x1 == x2:  
        return abs(px - x1)

    num = abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1)
    den = np.sqrt(((y2 - y1) ** 2) + ((x2 - x1) ** 2)) + 1e-6
    return num / den

def filter_close_lines(lines, angle_threshold=30, distance_threshold=10):
    lines = np.array(lines)
    removed_indices = set()

    angles = [np.degrees(np.arctan2(y2 - y1, x2 - x1)) if x1 != x2 else 90 for x1, y1, x2, y2 in lines]

    for i in range(len(lines)):
        if i in removed_indices:
            continue

        x1_i, y1_i, x2_i, y2_i = lines[i]
        theta_i = angles[i]

        for j in range(i + 1, len(lines)):
            if j in removed_indices:
                continue

            x1_j, y1_j, x2_j, y2_j = lines[j]
            theta_j = angles[j]

            if abs(theta_i - theta_j) < angle_threshold or (180 - abs(theta_i - theta_j)) < angle_threshold:
                d1 = point_to_line_distance(x1_i, y1_i, x1_j, y1_j, x2_j, y2_j)
                d2 = point_to_line_distance(x2_i, y2_i, x1_j, y1_j, x2_j, y2_j)
                max_dist = max(d1, d2)
                
                if max_dist < distance_threshold:
                    removed_indices.add(j)

    filtered_lines = [lines[i] for i in range(len(lines)) if i not in removed_indices]

    return filtered_lines

def compute_lane_angles(lane_segments):
    return [np.degrees(np.arctan2(y2 - y1, x2 - x1)) for x1, y1, x2, y2 in lane_segments]

def classify_situation(angles):
    if not angles:
        return 0  # No Lane Detected
    
    if len(angles) == 1:
        return 1  # Single Lane Road
    
    angle_diffs = [abs(a1 - a2) for i, a1 in enumerate(angles) for a2 in angles[i+1:]]

    if all(diff < 10 for diff in angle_diffs):
        return 2  # Multi-Lane Straight Road
    elif any(45 <= diff <= 135 for diff in angle_diffs):
        return 3  # Intersection Detected
    elif any(15 <= diff <= 35 for diff in angle_diffs):
        return 4  # Lane Merging/Splitting
    else:
        return 2  # Default to Multi-Lane Straight Road


def transform_ego(ego_locs, locs, oris, bbox, typs, ego_ori, T=11):
    ego_loc = ego_locs[0]

    keys = sorted(list(locs.keys()))
    locs = np.array([locs[k] for k in keys]).reshape(-1, T, 2)
    oris = np.array([oris[k] for k in keys]).reshape(-1, T)
    bbox = np.array([bbox[k] for k in keys]).reshape(-1, T, 2)
    typs = np.array([typs[k] for k in keys]).reshape(-1, T)

    R = [[np.sin(ego_ori), np.cos(ego_ori)], [-np.cos(ego_ori), np.sin(ego_ori)]]

    locs = (locs - ego_loc) @ R
    ego_locs = (ego_locs - ego_loc) @ R
    oris = oris - ego_ori

    return ego_locs, locs, oris, bbox, typs


class FSDataset(BasicDataset):
    def __init__(self, config_path, is_train=True):
        super().__init__(config_path, is_train=is_train)
        self.max_other = 10
        self.is_train = is_train
        if is_train:
            self.stage = "train"
        else:
            self.stage = "val"
        self.max_vehicle_radius = 100
        self.max_pedestrian_radius = 100
        self.num_plan = 20
        self.angles = []

    def __len__(self):
        return super().__len__()

    def get_prev_bevs(self, lmdb_txn, index):
        context = self.num_prev_timesteps
        prev_bevs = []
        for t, i in enumerate(range(index - context, index)):
            if i < 0:
                prev_bevs.append(np.zeros((4, 320, 320)))
                continue
            bev = self.load_bev(lmdb_txn, i, channels=[0, 1, 2, 6])
            bev = (bev > 0).astype(np.uint8).transpose(2, 0, 1)
            prev_bevs.append(bev)

        return prev_bevs

    def get_fs_segment(self, bevs, img, ego_locs, ego_oris, other_locs, cam_yaw=0, idx=None, cmd=None):
        PIXELS_PER_METER = 4
        PIXELS_AHEAD_VEHICLE = 120
        img_h, img_w, _ = img.shape
        fs_mask = np.zeros((img_h, img_w)).astype(np.uint8)
        other_locs = other_locs[1:]
        other_locs_center = other_locs * PIXELS_PER_METER
        other_locs_center[:, 1] -= PIXELS_AHEAD_VEHICLE
        other_locs_center = 320/2 - other_locs_center
        
        x_original = np.linspace(0, 1, ego_locs.shape[0])
        x_new = np.linspace(0, 1, 50)

        ego_locs = np.column_stack([
            np.interp(x_new, x_original, ego_locs[:, i]) for i in range(ego_locs.shape[1])
        ])
        ego_oris = np.interp(x_new, x_original, ego_oris)
                
        x = (ego_locs[:, 0]).tolist()
        y = ego_locs[:, 1].tolist()

        other_locs_raw = other_locs
        ego_locs_raw = ego_locs
        
        # print(other_locs)
        all_bboxes_in_bev = []
        for i, (px, py) in enumerate(zip(x, y)):
            px, py = int(320/2 - px * PIXELS_PER_METER), int(320/2 - py * PIXELS_PER_METER + PIXELS_AHEAD_VEHICLE)
            width = 6
            height = 8
            orientation = -ego_oris[i] + np.pi/2 # * 180 / np.pi
            l, t, r, b = px - width//2, py - height//2, px + width//2, py + height//2
            R = np.array([[np.cos(orientation), -np.sin(orientation)],
                          [np.sin(orientation), np.cos(orientation)]])
            points = np.array([(-width//2, height//2), (width//2, height//2), (width//2, -height//2), (-width//2, -height//2)])
            rotated_points = (R @ points.T).T
            rotated_points[:, 1] *= -1
            rotated_points += (px, py)
            
            other_locs = other_locs_center - (px, py)
            other_locs[:, 1] *= -1
            other_locs = (R.T @ other_locs.T).T
            
            mask = (other_locs[:, 0] <= width//2) * (other_locs[:, 0] >= -width//2) * (other_locs[:, 1] <= height//2) * (other_locs[:, 1] >= -height//2)
            if(i == 0):
                sign = np.ones(len(other_locs))
                sign[other_locs_raw[:, 1] < 0] = -1
                distance = np.sqrt(np.sum(other_locs ** 2, -1)) * sign
                continue
                
            if(mask.sum()):
                break
            
            rotated_points = rotated_points.astype(np.int32)
            x_z = np.concatenate([320/2 - rotated_points[:, 0:1], 320/2 + PIXELS_AHEAD_VEHICLE - rotated_points[:, 1:2]], 1) / PIXELS_PER_METER
            all_bboxes_in_bev.append(x_z)
        
        valid = False
        if(len(all_bboxes_in_bev) == 0):
            distance = distance[distance>0]
            if(len(distance)):
                if(distance.min() < 30):
                    valid = True
        else:
            centers = np.array(all_bboxes_in_bev).mean(1)
            diff_vec = centers[-1] - centers[0]
            distance = distance[distance>0]
            if(np.linalg.norm(diff_vec) < 1e-3):
                if(len(distance)):
                    if(distance.min() < 30):
                        valid = True
            else:
                valid = True

        angles = []
        try:
            centers = np.array(all_bboxes_in_bev).mean(1)
            diff_vec = centers[-1] - centers[0]
            angle = np.arctan2(diff_vec[1], diff_vec[0])
            self.angles.append(diff_vec)
        except:
            angle = np.array([0, 0])
            self.angles.append(angle)

        fov = 60
        rgb_w = 256
        rgb_h = 288
        focal = rgb_w / (2.0 * np.tan(fov * np.pi / 360.0))
        K = np.identity(3)
        K[0, 0] = K[1, 1] = focal
        K[0, 2] = rgb_w / 2.0
        K[1, 2] = rgb_h / 2.0
       
        for box_in_bev in all_bboxes_in_bev:
            ego_locs = box_in_bev
            ego_locs_xyz = np.concatenate([np.zeros((ego_locs.shape[0], 1)) - 2.5, ego_locs], -1)
            ego_locs_xyz = np.concatenate([-ego_locs_xyz[:, 1:2], -ego_locs_xyz[:, 0:1], ego_locs_xyz[:, 2:3]], -1)
        
            cam_rot = np.array([[np.cos(cam_yaw), 0, np.sin(cam_yaw)], [0, 1, 0], [-np.sin(cam_yaw), 0, np.cos(cam_yaw)]])
            ego_locs_xyz = (cam_rot @ ego_locs_xyz.T).T
            
            valid_mask = ego_locs_xyz[:, -1] != 0
            ego_locs_xyz[valid_mask] = ego_locs_xyz[valid_mask] / ego_locs_xyz[valid_mask, -1:]
            # ego_locs_xyz[:, :] = ego_locs_xyz[:, :] / ego_locs_xyz[:, -1:]
            proj_points = K @ ego_locs_xyz.T
            proj_points = (proj_points.T).astype(np.int32)
            
            if(np.all(proj_points[:, 1] > 0)):
                cv2.drawContours(fs_mask, [proj_points[:, :-1]], 0, (1, 1, 1), -1)

        return fs_mask.astype(np.float32), valid, angle,all_bboxes_in_bev
    
    def filter_sem(self, sem, labels=[7]):
        h, w = sem.shape
        resem = np.zeros((h, w, len(labels))).astype(sem.dtype)
        for i, label in enumerate(labels):
            resem[sem == label, i] = 1

        return resem

    def load_contour(self, contour):
        contour = contour[0]
        mask = (contour[:, 0] == -2) & (contour[:, 1] == -2)
        contour[mask] = np.tile(
            np.array([0, 1])[None], (mask.sum(), 1)
        )  # (bottom edge center)
        return contour[None]

    def __getitem__(self, idx):
        lmdb_txn = self.txn_map[idx]
        index = self.idx_map[idx]

        rgb2 = self.load_img(lmdb_txn, "rgb_2", index)
        rgb3 = self.load_img(lmdb_txn,"rgb_3",index)
        rgb1 = self.load_img(lmdb_txn,"rgb_1",index)
        
        sem1 = self.load_img(lmdb_txn, "sem_1", index)
        sem2 = self.load_img(lmdb_txn, "sem_2", index)
        sem3 = self.load_img(lmdb_txn, "sem_3", index)
    
        # SITUATION
        channels = list(range(11))
        bev_key=self.load_bev(lmdb_txn,index,channels=channels)
        binary_mask = bev_key[:, :, 10].astype(np.uint8)
        combined_bev = np.stack([binary_mask * 255] * 3, axis=-1)  # Shape: (310, 320, 3)

        ego_x, ego_y = 160, 300
        roi_width, roi_height = 100, 100

        roi, (x_min, y_min, x_max, y_max) = extract_roi_above_ego(combined_bev, ego_x, ego_y, roi_width, roi_height)
        lane_segments = extract_lane_segments_from_roi(roi)
        H = 100  # ROI IMAGE Height
        lane_segments = convert_to_cartesian(lane_segments, H)
        filtered_segments = filter_close_lines(lane_segments)
        angles = compute_lane_angles(filtered_segments)
        situation = classify_situation(angles)

        bev = self.load_bev(lmdb_txn, index, channels=[0, 1, 2, 6])
        bev = (bev > 0).astype(np.uint8).transpose(2, 0, 1)
        prev_bevs = self.get_prev_bevs(lmdb_txn, index)
        prev_bevs.append(bev)
        bevs = np.stack(prev_bevs)


        # Vehicle locations/orientations
        ego_id, ego_locs, ego_oris, ego_bbox, msks, locs, oris, bbox, typs = (
            self.filter(
                lmdb_txn,
                index,
                max_pedestrian_radius=self.max_pedestrian_radius,
                max_vehicle_radius=self.max_vehicle_radius,
                T=self.num_plan,
            )
        )

        ego_locs, locs, oris, bbox, typs = transform_ego(
            ego_locs, locs, oris, bbox, typs, ego_oris[0], self.num_plan + 1
        )

        cmd = int(self.access("cmd", lmdb_txn, index, 1, dtype=np.uint8))
        nxp = self.access("nxp", lmdb_txn, index, 1).reshape(2)

        bbox_other = np.zeros((self.max_other, self.num_plan + 1, 2))
        oris_other = np.zeros((self.max_other, self.num_plan + 1))
        locs_other = np.zeros((self.max_other, self.num_plan + 1, 2))
        mask_other = np.zeros(self.max_other)
        bbox_other[: bbox.shape[0]] = bbox[: self.max_other]
        locs_other[: locs.shape[0]] = locs[: self.max_other]
        oris_other[: oris.shape[0]] = oris[: self.max_other]
        mask_other[: len(bbox)] = 1

        ret = (
            bevs.astype(np.float32),
            ego_locs.astype(np.float32),
            oris,
            rgb2,
            locs_other.astype(np.float32),
            mask_other.astype(np.bool_),
        )
        bevs = ret[0]
        
        ego_locs = ret[1]
        ego_oris = ret[2][0]
        mask_other = ret[5]
        other_locs = ret[4][mask_other][:, 0, :]  # (N, 2)

        mask3, valid, angle,all_bboxes_in_bev = self.get_fs_segment(bevs, rgb3, ego_locs, ego_oris, other_locs, cam_yaw=np.radians(-60), idx=idx, cmd=cmd)
        mask2, valid, angle,all_bboxes_in_bev = self.get_fs_segment(bevs, rgb2, ego_locs, ego_oris, other_locs, cam_yaw=np.radians(0), idx=idx, cmd=cmd)
        mask1, valid, angle,all_bboxes_in_bev = self.get_fs_segment(bevs, rgb1, ego_locs, ego_oris, other_locs, cam_yaw=np.radians(60), idx=idx, cmd=cmd)
        
        all_bboxes_bev = np.array(all_bboxes_in_bev)
        
        mask1[mask1 > 0] = 1
        mask2[mask2 > 0] = 1
        mask3[mask3 > 0] = 1
        
        img_ = np.hstack((rgb1, rgb2, rgb3))
        img_ = img_.transpose(2, 0, 1).astype(np.float32)
        mask_ = np.hstack((mask1, mask2, mask3))
        mask_[mask_ == -2] = 0
        mask_ = (mask_ * 255).astype(np.uint8) 
        contours, _ = cv2.findContours(mask_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:            
            contour = np.ones((50, 2), dtype=np.float32) * -2
            contour_key = False
        else:
            max_index = np.array([len(c) for c in contours]).argmax()
            contour = contours[max_index]
            contour = self.interpolate_contour_points(contour[:, 0], 50)
            contour[:,0] = contour[:,0]/img_.shape[2]
            contour[:,1] = contour[:,1]/img_.shape[1]
            contour = contour*2 - 1 # between -1 and 1
            contour_key = True

        contour = self.load_contour(contour[None]).astype(np.float32)
        data = {"img1": rgb1, "img2": rgb2, "img3": rgb3, "mask1": mask1, "mask2": mask2, "mask3": mask3, "contour": contour, "contour_key": contour_key}
        data["valid"] = valid
        data["angle"] = angle
        data["ego_locs"] = ego_locs
        
        data["cmd"] = cmd
        data["idx"] = idx
        data["sem1"] = sem1
        data["sem2"] = sem2
        data["sem3"] = sem3
        data['situation'] = situation
        # data["bev"] = bev_key
        
        # data["all_bboxes_bev"] = all_bboxes_bev

        return data
    
    def interpolate_contour_points(self, contour, target_num_points):
        num_points = len(contour)

        if not np.array_equal(contour[0], contour[-1]):
            contour = np.vstack([contour, contour[0]]) 
            num_points += 1

        distances = np.cumsum(np.sqrt(np.sum(np.diff(contour, axis=0) ** 2, axis=1)))
        distances = np.insert(distances, 0, 0)

        normalized_distances = distances / distances[-1]

        target_distances = np.linspace(0, 1, target_num_points, dtype=np.float32)
        contour_interp = np.zeros((target_num_points, 2), dtype=np.float32)
        for i in range(2):
            contour_interp[:, i] = np.interp(
                target_distances, normalized_distances, contour[:, i]
            )

        return contour_interp

def moving_average(points, window_size=5):
    kernel = np.ones(window_size) / window_size
    x_smooth = np.convolve(points[:, 0], kernel, mode='same')
    y_smooth = np.convolve(points[:, 1], kernel, mode='same')
    return np.column_stack((x_smooth, y_smooth))


if __name__ == "__main__":

    dataset = FSDataset("lav.yaml", is_train=False)
    dataloader = torch.utils.data.dataloader.DataLoader(
        dataset, batch_size=1, num_workers=16, shuffle=False
    )
    
    output_folder = "/ssd_scratch/cvit/keshav/carla_cached_data/"
    os.makedirs(output_folder, exist_ok=True)

    count = {}
    for i, batch in tqdm(enumerate(dataloader)):
        new = {}
        for k, v in batch.items():
            new[k] = v.numpy()[0]
        np.savez(f"{output_folder}/{i}.npz", **new)

    # Uncomment for visualizing
    
    # for i in tqdm(range(1600, len(dataset))):
    #     data = dataset[i]
    #     valid = data['valid']
    #     cmd = data['cmd']

    #     img1 = data["img1"]
    #     img2 = data["img2"]
    #     img3 = data["img3"]
    #     img = np.hstack((img1, img2, img3))
        
    #     mask1 = data["mask1"]
    #     mask2 = data["mask2"]
    #     mask3 = data["mask3"]
        
    #     situation = data['situation']
    #     mask = np.hstack((mask1, mask2, mask3))
    #     mask[mask == -2] = 0
        
    #     mask = mask * 255
    #     mask = mask.astype(np.uint8)
        
    #     overlay = np.copy(img)
    #     overlay[mask == 255] = [0, 255, 255]

    #     if(valid):
    #         cv2.imwrite(f"/ssd_scratch/cvit/keshav/carla_vis/img_{situation}_{i}.png", overlay)