
"""
In datasets/oxford_robotcar there should be the following files/folders:
gps, stereo/centre
(datasets is a folder that contains files such as pitts250k_train.mat)
"""

import json
import os
from tqdm import tqdm
import numpy as np
import subprocess
import pandas

import util
import map_builder
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--new_dataset_dir', type=str, required=True)
parser.add_argument('--raw_dataset_dir', type=str, required=True)
parser.add_argument('--subset_name', type=str, required=True)
args = parser.parse_args()

ds_new = args.new_dataset_dir
ds_raw = args.raw_dataset_dir
ds_name = args.subset_name

# download routes from odoviz v2 after loading and processing
# each processed route should contain poses.csv
data_extra_dir = 'data/oxford_robotcar'
route_1 = 'db_selected.csv'  # summer (day)
route_2 = 'q_selected.csv'  # winter (day)

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


def get_route_name(pose_file):
    with open(pose_file, 'r') as f:
        line = f.readline()
        while line.startswith("#"):
            tokens = line[1:].split(':')
            tokens = list(map(lambda x: x.strip(), tokens))
            if tokens[0].lower() == 'route':
                route_name = tokens[1]
                return route_name
            line = f.readline()

# Database
g_pose_file = os.path.join(data_extra_dir, ds_name, route_1)
g_pose_df = pandas.read_csv(g_pose_file, comment="#")
g_route_name = get_route_name(g_pose_file)
# Queries
q_pose_file = os.path.join(data_extra_dir, ds_name, route_2)
q_pose_df = pandas.read_csv(q_pose_file, comment="#")
q_route_name = get_route_name(q_pose_file)

assert len(g_pose_df) == len(q_pose_df)

# split into train, val, test
# 80% train + val, 20% test
# of 80% train, 80% train, 20% val
num_train_val = int(len(g_pose_df)*0.8)
idxs = {}
idxs['train'] = list(range(0, int(num_train_val*0.80)))
idxs['val'] = list(range(idxs['train'][-1], num_train_val))
idxs['test'] = list(range(idxs['val'][-1], len(g_pose_df)))

# move exceptions
exceptions = {
    'sd2wd': { 'train': [(1418134252.549121, 1418134487.454799)] },
    'wd2wn': { 'nil': [
        (1418237403.122886, 1418237488.0488539),
        (1418237304.573770, 1418237310.198005),
        (1418236613.542741, 1418236619.291957),
    ] }
}
for e_name, e_values in exceptions.items():
    if ds_name != e_name:
        continue
    for subset, s_values in e_values.items():
        idxs_matched = set()
        for (t_min, t_max) in s_values:
            q_pose_df_filtered = q_pose_df.apply(lambda x: x['timestamp'] >= t_min and x['timestamp'] <= t_max, axis=1)
            idxs_matched.update(q_pose_df_filtered.index[q_pose_df_filtered].tolist())
        for s in idxs.keys():
            if s == subset:
                idxs[s] += list(idxs_matched)
            else:
                idxs[s] = [i for i in idxs[s] if i not in idxs_matched]

for dataset in ["train", "val", "test"]:
    # extract img from json in csv
    g_pose_df_ds = g_pose_df.iloc[idxs[dataset]]
    g_images = g_pose_df_ds['imgs'].apply(lambda x: json.loads(x)['stereoCentre']).tolist()
    g_images = [os.path.join(g_route_name, 'stereo', 'centre', img) for img in g_images]
    g_lat_longs = g_pose_df_ds[['latitude', 'longitude']].values
    g_pitch_yaws = g_pose_df_ds[['pitch', 'yaw']].values
    g_dst = os.path.join(ds_new, 'images', dataset, 'database')
    copy_images(g_dst, g_images, g_lat_longs, g_pitch_yaws, f"{dataset} db images")

    q_pose_df_ds = q_pose_df.iloc[idxs[dataset]]
    q_images = q_pose_df_ds['imgs'].apply(lambda x: json.loads(x)['stereoCentre']).tolist()
    q_images = [os.path.join(q_route_name, 'stereo', 'centre', img) for img in q_images]
    q_lat_longs = q_pose_df_ds[['latitude', 'longitude']].values
    q_pitch_yaws = q_pose_df_ds[['pitch', 'yaw']].values
    q_dst = os.path.join(ds_new, 'images', dataset, 'queries')
    copy_images(q_dst, q_images, q_lat_longs, q_pitch_yaws, f"{dataset} query images")

print('\nCreating map...')
map_builder.build_map_from_dataset(ds_new)
# shutil.rmtree(raw_data_folder)
