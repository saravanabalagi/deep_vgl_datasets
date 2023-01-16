
"""
In datasets/oxford_robotcar there should be the following files/folders:
gps, stereo/centre
(datasets is a folder that contains files such as pitts250k_train.mat)
"""

import json
import os
from tqdm import tqdm
import subprocess
import pandas

import util
import map_builder
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--new_dataset_dir', type=str, required=True)
parser.add_argument('--raw_dataset_dir', type=str, required=True)
args = parser.parse_args()

ds_new = args.new_dataset_dir
ds_raw = args.raw_dataset_dir

# download routes from odoviz v2 after loading and processing
# each processed route should contain poses.csv
data_extra_dir = 'data/oxford_robotcar'
route_1 = '2015-05-19-14-06-38'  # summer (day)
route_2 = '2014-12-09-13-21-02'  # winter (day)

if not os.path.exists(ds_raw):
    raise ValueError(f"raw_datasets_dir {ds_raw} does not exist.")
os.makedirs(ds_new, exist_ok=True)

def copy_images(dst_folder, src_images_paths, lat_longs, pitch_yaws, tqdm_desc):
    os.makedirs(dst_folder, exist_ok=True)
    for i, (src_image_path, (lat, long), (pitch, yaw)) in enumerate(zip(tqdm(src_images_paths, desc=f"Copying {tqdm_desc}", ncols=100),
                                                     lat_longs, pitch_yaws)):
        src_image_name = os.path.basename(src_image_path)
        src_image_name_without_ext = os.path.splitext(src_image_name)[0]
        tile_num = pitch*24 + yaw
        dst_image_name = util.get_dst_image_name(lat, long, pano_id=i,
                                             tile_num=tile_num, note=src_image_name_without_ext)
        src_path = os.path.join(ds_raw, src_image_path)
        dst_path = os.path.join(dst_folder, dst_image_name)

        # create symlink
        cmd = "ln -s {} {}".format(src_path, dst_path)
        subprocess.call(cmd, shell=True)
        # shutil.move(src_path, dst_path)

# Database
g_pose_file = os.path.join(data_extra_dir, route_1, "gps", f"poses.csv")
g_pose_df = pandas.read_csv(g_pose_file, comment="#")
# Queries
q_pose_file = os.path.join(data_extra_dir, route_1, "gps", f"poses.csv")
q_pose_df = pandas.read_csv(q_pose_file, comment="#")

assert len(g_pose_df) == len(q_pose_df)

# split into train, val, test
# 80% train + val, 20% test
# of 80% train, 70% train, 30% val
num_train_val = int(len(g_pose_df)*0.8)
idxs = {}
idxs["train"] = (0, int(num_train_val*0.7))
idxs["val"] = (idxs["train"][1], num_train_val)
idxs["test"] = (idxs["val"][1], len(g_pose_df))

for dataset in ["train", "val", "test"]:
    # extract img from json in csv
    g_pose_df_ds = g_pose_df.iloc[idxs[dataset][0]:idxs[dataset][1]]
    g_images = g_pose_df_ds['imgs'].apply(lambda x: json.loads(x)['stereoCentre']).tolist()
    g_lat_longs = g_pose_df_ds[['latitude', 'longitude']].values
    g_pitch_yaws = g_pose_df_ds[['pitch', 'yaw']].values
    g_dst = os.path.join(ds_new, 'images', dataset, 'database')
    copy_images(g_dst, g_images, g_lat_longs, g_pitch_yaws, f"{dataset} db images")

    q_pose_df_ds = q_pose_df.iloc[idxs[dataset][0]:idxs[dataset][1]]
    q_images = q_pose_df_ds['imgs'].apply(lambda x: json.loads(x)['stereoCentre']).tolist()
    q_lat_longs = q_pose_df_ds[['latitude', 'longitude']].values
    q_pitch_yaws = q_pose_df_ds[['pitch', 'yaw']].values
    q_dst = os.path.join(ds_new, 'images', dataset, 'queries')
    copy_images(q_dst, q_images, q_lat_longs, q_pitch_yaws, f"{dataset} query images")

map_builder.build_map_from_dataset(ds_new)
# shutil.rmtree(raw_data_folder)
