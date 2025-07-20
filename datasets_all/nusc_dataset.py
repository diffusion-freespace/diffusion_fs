import numpy as np
import glob
import cv2
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class NuScenesDataset(Dataset):
    def __init__(self, dataroot, is_train=True):
        self.img_size = (512, 256) # (width, height)
        self.dataroot = dataroot
        self.num_contour_points = 50
        self.is_train = is_train
        if(is_train):
            self.fs_masks = glob.glob(dataroot + "/samples/FS/*_train.jpg")
            self.cam_front_images = [f.replace("/FS/", "/CAM_FRONT/")[:-10] + ".jpg" for f in self.fs_masks]
            self.meta_data = [f.replace("/samples/FS/", "/fs_meta/").split(".")[0] + ".npz" for f in self.fs_masks]
            self.num_images = len(self.fs_masks)
        else:
            self.fs_masks = glob.glob(dataroot + "/samples/FS/*_val.jpg")
            self.cam_front_images = [f.replace("/FS/", "/CAM_FRONT/")[:-8] + ".jpg" for f in self.fs_masks]
            self.meta_data = [f.replace("/samples/FS/", "/fs_meta/").split(".")[0] + ".npz" for f in self.fs_masks]
            self.num_images = len(self.fs_masks)
        self.count = 0
        
    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):    
        img = cv2.imread(self.cam_front_images[idx])/255
        img = cv2.resize(img, self.img_size)
        fs_mask = cv2.imread(self.fs_masks[idx], cv2.IMREAD_GRAYSCALE)
        fs_mask[fs_mask >= 1] = 1
        fs_mask[fs_mask < 1] = 0
        fs_mask = cv2.resize(fs_mask, self.img_size)
        
        contours, hierarchy = cv2.findContours(fs_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        all_contours = []
        
        all_len = []
        for contour in contours:
            contour_interp = self.interpolate_contour_points(contour[:, 0], self.num_contour_points)
            all_len.append(len(contour[:, 0]))
            all_contours.append(contour_interp)
        
        valid = True
        if(len(all_contours) == 0):
            contour = np.ones((self.num_contour_points, 2)) * -2
            valid = False
        else:
            max_index = np.array(all_len).argmax()
            contour = all_contours[max_index]
            contour[:, 0] = contour[:, 0] / self.img_size[0]
            contour[:, 1] = contour[:, 1] / self.img_size[1]
            contour = contour * 2 - 1
            
            # uncomment for visualizing contours
            
            # output_image = img.copy()
            # for point in contour:
            #     x = ((point + 1) * self.img_size[0] / 2).astype(np.int32)[0]
            #     y = ((point + 1) * self.img_size[1] / 2).astype(np.int32)[1]
            #     cv2.circle(output_image, (x, y), radius=1, color=(1, 0, 0), thickness=3)
            # cv2.imwrite(f"/ssd_scratch/cvit/keshav/nusc_samples_vis/{idx}.png", output_image * 255)

        img = img.transpose(2, 0, 1).astype(np.float32)
        data = {"img" : img * 255, "mask": fs_mask, "contour": contour[None].astype(np.float32)}
        data['valid'] = valid
        data['idx'] = idx
        data['img_path'] = self.cam_front_images[idx]
        
        if(self.is_train == False):
            meta_data = np.load(self.meta_data[idx])
            obstacle_mask = meta_data['obstacle_mask']
            drivable_area_mask = meta_data['drivable_area_mask'].astype(np.uint8)
            obstacle_mask = cv2.resize(obstacle_mask, self.img_size)
            drivable_area_mask = cv2.resize(drivable_area_mask, self.img_size)
            obs_masks = np.tile(obstacle_mask[..., None], (1, 1, 2))
            sem = np.concatenate([obs_masks, drivable_area_mask[..., None]], -1).transpose(2, 0, 1).astype(np.float32)
            data['obstacles'] = sem
        
        return data
    
    def interpolate_contour_points(self, contour, target_num_points):
        num_points = len(contour)
        indices = np.linspace(0, num_points - 1, target_num_points, dtype=np.float32)
        
        contour_interp = np.zeros((target_num_points, 2), dtype=np.float32)
        for i in range(2):
            contour_interp[:, i] = np.interp(indices, np.arange(num_points), contour[:, i])
        
        return contour_interp


if __name__ == "__main__":
    dataset = NuScenesDataset("/ssd_scratch/cvit/keshav/nuscenes/", is_train=False)
    dataLoader = DataLoader(dataset, batch_size=64)
    for batch in tqdm(dataLoader):
        batch