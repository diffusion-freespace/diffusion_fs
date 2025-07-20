import numpy as np
import cv2
import torch
import os
import glob
import yaml
from torch.utils.data import Dataset

class FSCacheDataset(Dataset):
    def __init__(self, config_path, is_train=True):
        super().__init__()
        self.config_path = config_path
        self.is_train = is_train
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        for key, value in config.items():
            setattr(self, key, value)
        
        self.data_dir = self.data_dir_train if is_train else self.data_dir_test
        self.files = sorted(glob.glob(self.data_dir + "/*.npz"))
            
        if is_train:
            self.stage = "train"
        else:
            self.stage = "val"

    def __len__(self):
        return len(os.listdir(self.data_dir))

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
        # cache_path = f"{self.data_dir}/{idx}.npz"
        cache_path = self.files[idx]

        if os.path.exists(cache_path):
            new_data = {}
            data = dict(np.load(cache_path))
            mask1 = data["mask1"]
            mask2 = data["mask2"]
            mask3 = data["mask3"]
            
            img1 = data["img1"]
            img2 = data["img2"]
            img3 = data["img3"]
        
            img = np.hstack((img1, img2, img3))
            img = img.transpose(2, 0, 1).astype(np.float32)
            
            mask = np.hstack((mask1, mask2, mask3))
            mask[mask == -2] = 0
        
            new_data["img"] = img
            mask = mask.astype(np.uint8)
            
            if("contour" not in data):
                contours, _ = cv2.findContours(mask * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if len(contours) == 0:
                    contour = np.ones((50, 2), dtype=np.float32) * -2
                else:
                    max_index = np.array([len(c) for c in contours]).argmax()
                    contour = contours[max_index]
                    contour = self.interpolate_contour_points(contour[:, 0], 50)
                    contour[:, 0] = contour[:, 0] / img.shape[2]
                    contour[:, 1] = contour[:, 1] / img.shape[1]
                    contour = contour * 2 - 1
                
                contour = self.load_contour(contour[None]).astype(np.float32)
                data["contour"] = contour

            new_data["contour"] = data["contour"]

            if("situation" in data):
                new_data["situation"] = data["situation"]
            
            new_data["valid"] = data["valid"]
            new_data["cmd"] = data["cmd"]
            
            if self.stage != "train":
                new_data["mask"] = mask

                sem1 =  data["sem1"]
                sem2 = data["sem2"]
                sem3 = data["sem3"]
                sem = np.hstack((sem1, sem2, sem3))
                
                if(sem.ndim == 2):
                    sem = self.filter_sem(
                        sem, labels=[4, 10, 7]
                    )  # ped, cars, road_markings

                sem = sem.transpose(2, 0, 1)
                new_data["obstacles"] = sem
            return new_data
        
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


if __name__ == "__main__":
    dataset = FSCacheDataset("lav.yaml", is_train=False)
    dataloader = torch.utils.data.dataloader.DataLoader(
        dataset, batch_size=1, num_workers=16, shuffle=False
    )
    print(dataset[0].keys())