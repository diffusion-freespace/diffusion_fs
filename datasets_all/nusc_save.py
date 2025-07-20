import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# from nuscenes.nuscenes import NuScenes
from nuscenes import NuScenes

from pyquaternion import Quaternion
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.map_expansion import arcline_path_utils
from nuscenes.map_expansion.bitmap import BitMap
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.utils.splits import create_splits_scenes
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import Polygon
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from scipy.spatial import ConvexHull
from skimage.draw import polygon
import cv2
from matplotlib import pyplot as plt
import descartes
import os

class NuScenesDataset(Dataset):
    def __init__(self, dataroot):
        self.dataroot = dataroot
        self.nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True)
        self.locations = ['singapore-onenorth', 'singapore-hollandvillage', 'singapore-queenstown', 'boston-seaport']
        self.nusc_map_dict = {map_name : NuScenesMap(dataroot=dataroot, map_name = map_name) for map_name in self.locations}
        self.nusc_can = NuScenesCanBus(dataroot=dataroot)
        
        self.train_scenes = self.get_scenes(0)
        self.val_scenes = self.get_scenes(1)
        self.test_scenes = self.get_scenes(2)
        os.makedirs(f"{dataroot}/fs_meta/", exist_ok=True)
        
    def __len__(self):
        return len(self.nusc.sample)
    
    def get_lanes(self, sample, location):
        layer_names = ['drivable_area', 'lane', 'ped_crossing', 'walkway', 'stop_line', 'road_segment', 'road_block', 'carpark_area']
        scene_record = self.nusc.get('scene', sample['scene_token'])
        cam_token = sample['data']['CAM_FRONT']
        cam_record = self.nusc.get('sample_data', cam_token)
        cs_record = self.nusc.get('calibrated_sensor', cam_record['calibrated_sensor_token'])
        cam_intrinsic = np.array(cs_record['camera_intrinsic'])
        cam_path = self.nusc.get_sample_data_path(cam_token)
        im = Image.open(cam_path)
        im_size = im.size
        mask = np.zeros((im_size[1], im_size[0]))

        poserecord = self.nusc.get('ego_pose', cam_record['ego_pose_token'])
        ego_pose = poserecord['translation']
        patch_radius = 1000
        box_coords = (
            ego_pose[0] - patch_radius,
            ego_pose[1] - patch_radius,
            ego_pose[0] + patch_radius,
            ego_pose[1] + patch_radius,
        )
        
        records_in_patch = self.nusc_map_dict[location].explorer.get_records_in_patch(box_coords, layer_names, 'intersect')
        near_plane = 1e-8
        
        all_lanes = []

        for layer_name in layer_names:
            for token in records_in_patch[layer_name]:
                record = self.nusc_map_dict[location].explorer.map_api.get(layer_name, token)
                # print(record.keys())
                if layer_name == 'drivable_area':
                    polygon_tokens = record['polygon_tokens']
                else:
                    polygon_tokens = [record['polygon_token']]

                # print(len(polygon_tokens))
                for polygon_token in polygon_tokens:
                    polygon = self.nusc_map_dict[location].explorer.map_api.extract_polygon(polygon_token)
                    # if(polygon.contains(Point(ego_pose[0], ego_pose[1]))):
                    #     color = self.color_map['lane']
                    # else:
                    #     color = self.color_map['ped_crossing']

                    # Convert polygon nodes to pointcloud with 0 height.
                    points = np.array(polygon.exterior.xy)
                    # print(points)
                    # exit(0)
                    points = np.vstack((points, np.zeros((1, points.shape[1]))))

                    # Transform into the ego vehicle frame for the timestamp of the image.
                    points = points - np.array(poserecord['translation']).reshape((-1, 1))
                    points = np.dot(Quaternion(poserecord['rotation']).rotation_matrix.T, points)

                    # Transform into the camera.
                    points = points - np.array(cs_record['translation']).reshape((-1, 1))
                    points = np.dot(Quaternion(cs_record['rotation']).rotation_matrix.T, points)

                    # Remove points that are partially behind the camera.
                    depths = points[2, :]
                    behind = depths < near_plane
                    if np.all(behind):
                        continue

                    points = NuScenesMapExplorer._clip_points_behind_camera(points, near_plane)

                    # Ignore polygons with less than 3 points after clipping.
                    if len(points) == 0 or points.shape[1] < 3:
                        continue

                    # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
                    points = view_points(points, cam_intrinsic, normalize=True)
                    inside = np.ones(points.shape[1], dtype=bool)
                    # inside = np.logical_and(inside, points[0, :] > 1)
                    inside = np.logical_and(inside, points[0, :] > -10000)
                    # inside = np.logical_and(inside, points[0, :] < im.size[0] - 1)
                    inside = np.logical_and(inside, points[0, :] < 20000)
                    inside = np.logical_and(inside, points[1, :] > 1)
                    # inside = np.logical_and(inside, points[1, :] < im.size[1] - 1)
                    inside = np.logical_and(inside, points[1, :] < 20000)
                    # prev_points = points
                    if np.all(np.logical_not(inside)):
                        continue
                    
                    points = points[:, inside]
                    # print(len(prev_points[:2].T) - len(points[:2].T))
                    points = points[:2, :].astype(np.int64).T
                    # points = points[:2, :]
                    # points = [(p0, p1) for (p0, p1) in zip(points[0], points[1])]
                    # all_lanes.append(np.array(points))
                    cv2.fillPoly(mask, [points], 1)
                    # polygon_proj = Polygon(points)
                    
                    # x, y = polygon_proj.exterior.xy 
                    # # rr, cc = polygon(np.array(y, dtype=np.int32), np.array(x, dtype=np.int32), shape=mask.shape)
                    # # mask[rr, cc] = 255
                    # polygon_proj = Polygon(points)
                    # ax.add_patch(descartes.PolygonPatch(polygon_proj, fc="#33a02c", alpha=0.3,
                                                        # label="lane"))
                    # # Filter small polygons
                    # if polygon_proj.area < min_polygon_area:
                    #     continue
                    
        return mask, np.asarray(im), cam_path
    
    def get_scenes(self, is_train):
        # filter by scene split
        split = {'v1.0-trainval': {0: 'train', 1: 'val', 2: 'test'},
                 'v1.0-mini': {0: 'mini_train', 1: 'mini_val'},}[
            self.nusc.version
        ][is_train]

        blacklist = [419] + self.nusc_can.can_blacklist  # # scene-0419 does not have vehicle monitor data
        blacklist = ['scene-' + str(scene_no).zfill(4) for scene_no in blacklist]

        scenes = create_splits_scenes()[split][:]
        for scene_no in blacklist:
            if scene_no in scenes:
                scenes.remove(scene_no)

        return scenes
    
    def get_bbox_mask(self, sample_token, camera_channel='CAM_FRONT'):
        sample = self.nusc.get('sample', sample_token)
        camera_token = sample['data'][camera_channel]
        data_path, boxes, camera_intrinsic = self.nusc.get_sample_data(camera_token)
        
        mask = np.zeros((900, 1600), dtype=np.uint8)
        mask_image = Image.fromarray(mask)
        draw = ImageDraw.Draw(mask_image)
        for box in boxes:
            corners_3d = box.corners()  # Shape: (3, 8)
            corners_2d = view_points(corners_3d, camera_intrinsic, normalize=True)[:2].T  # Shape: (8, 2)
            try:
                hull = ConvexHull(corners_2d)
                polygon = corners_2d[hull.vertices]
            except:
                continue  # Skip invalid/occluded boxes
            polygon = [(int(x), int(y)) for x, y in polygon]
            draw.polygon(polygon, fill=1)
            
        return mask_image

    def __getitem__(self, idx):
        sample_token = self.nusc.sample[idx]['token']
        
        sample = self.nusc.get("sample", sample_token)
        scene = self.nusc.get("scene", sample['scene_token'])
        location = self.nusc.get('log', scene['log_token'])['location']
        
        bbox_mask = self.get_bbox_mask(sample_token)
        bbox_mask = np.asarray(bbox_mask)
        
        drivable_area_mask, im, cam_path = self.get_lanes(self.nusc.get('sample', sample_token), location)
        drivable_area_mask = drivable_area_mask.astype(np.uint8).astype(np.bool_)
        
        camera_channel = 'CAM_FRONT'
        scene_token = sample["scene_token"]
        scene = self.nusc.get("scene", scene_token)
        scene_number = scene["name"]
        
        suffix = "train"
        if(scene_number in self.val_scenes):
            suffix = "val"
        elif (scene_number in self.test_scenes):
            suffix = "test"
        
        
        mask, img, cam_path, ego_poses = self.nusc.render_future_footprint(sample_token, camera_channel=camera_channel, min_polygon_area=100)
        filename = cam_path.split("/")[-1]
        mask_store = f"{self.dataroot}/samples/FS/{filename.split('.')[0]}_{suffix}.jpg"
        cv2.imwrite(mask_store, mask*255)
        
        np.savez(f"{self.dataroot}/fs_meta/{filename.split('.')[0]}_{suffix}.npz", obstacle_mask=bbox_mask, drivable_area_mask=drivable_area_mask)
        
        mask = np.zeros(1) # return something dummy just for the dataloader to collate something
        return mask

if __name__ == "__main__":
    
    dataset = NuScenesDataset("/ssd_scratch/cvit/keshav/nuscenes/")
    dataLoader = DataLoader(dataset, batch_size=64, num_workers=16)

    for batch in tqdm(dataLoader):
        batch
        # exit(0)