import numpy as np
import glob

all_exp = ["base", "obs", "cls", "noise", "obs_cls"]

for exp in all_exp:
    all_files = glob.glob(f"results/sit_*/{exp}.txt")
    all_iou_dict = {}
    all_obs_dict = {}
    all_road_dict = {}
    all_count_dict = {}
    for file in all_files:
        with open(file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if "road" in line:
                    val_road = float(line.split(":")[-1].strip())
                    all_road_dict[file.split('/')[-2].split('_')[-1]] = val_road

                if "iou" in line:
                    val_iou = float(line.split(":")[-1].strip())
                    all_iou_dict[file.split('/')[-2].split('_')[-1]] = val_iou
                    
                if "obs" in line:
                    val_obs = float(line.split(":")[-1].strip())
                    all_obs_dict[file.split('/')[-2].split('_')[-1]] = val_obs
                
                if "count" in line:
                    count = int(line.split(":")[-1].strip())
                    all_count_dict[file.split('/')[-2].split('_')[-1]] = count
                    
    all_iou = []
    all_counts = []
    all_obs = []
    all_road = []

    for k in sorted(all_iou_dict.keys()):
        all_counts.append(all_count_dict[k])
        all_iou.append(all_iou_dict[k])
        all_obs.append(all_obs_dict[k])
        all_road.append(all_road_dict[k])
        
    counts = np.array(all_counts)
    all_iou = np.array(all_iou)
    all_obs = np.array(all_obs)
    all_road = np.array(all_road)

    iou = (all_iou * counts).sum() / np.sum(counts)
    obs = (all_obs * counts).sum() / np.sum(counts)
    road = (all_road * counts).sum() / np.sum(counts)

    print(exp)
    print(f"Final IoU: {iou}")
    print(f"Final Obs IoU: {obs}")
    print(f"Final Road IoU: {road}")
    print()