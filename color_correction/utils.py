import os
import cv2
import json
import numpy as np
import math
from . import deep_wb_single_task
import torch
from . import deep_wb_blocks
from .utilities.deepWB import deep_wb
from PIL import Image

def load_data(dir_path):
    imgs = []
    pcds = []
    imus = []
    img_path = os.path.join(dir_path, 'camera')
    pcd_path = os.path.join(dir_path, 'lidar')
    imu_path = os.path.join(dir_path, 'supplement')
    for img_name in os.listdir(img_path):
        img = os.path.join(img_path, img_name)
        pcd = os.path.join(pcd_path, img_name.replace('.png', '.bin'))
        imu = os.path.join(imu_path, img_name.replace('.png', '.json'))
        img = cv2.imread(img)
        pcd = np.fromfile(pcd, dtype=np.float32).reshape(-1, 4)
        with open(imu, 'r') as f:
            imu = json.load(f)
        imgs.append(img)
        pcds.append(pcd)
        imus.append(imu)
    calib_path = os.path.join(dir_path, 'calib.json')
    with open(calib_path, 'r') as f:
        calib = json.load(f)
    return imgs, pcds, imus, calib

def project_lidar_to_image(pcd, calib):
    pcd = pcd[:, :3]
    pcd = np.hstack([pcd, np.ones((pcd.shape[0], 1))])
    intrinsic = np.array(calib['intrinsic']).reshape(3, 3)
    extrinsic = np.array(calib['extrinsic']).reshape(4, 4)
    pcd = extrinsic @ np.transpose(pcd)
    pcd = np.delete(pcd, 3, 0)
    pcd = intrinsic @ pcd
    pcd = pcd.T
    pcd = pcd[:, :2] / pcd[:, 2:]
    return pcd

def draw_lidar_on_image(img, pcd, calib):
    pcd = project_lidar_to_image(pcd, calib)
    pcd = pcd.astype(np.int32)
    mask = img.copy()
    for p in pcd:
        cv2.circle(mask, (p[0], p[1]), 5, (0, 0, 255), -1)
    img = cv2.addWeighted(img, 0.5, mask, 0.5, 0)
    return img

def auto_exposure(img, pcd_2d):
    '''
    Auto exposure correction based on the projected 2d lidar points
    '''
    pcd_2d = pcd_2d.astype(np.int32)
    pcd_2d = np.unique(pcd_2d, axis=0)
    img_pixels = img[pcd_2d[:, 1], pcd_2d[:, 0]]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_partial = cv2.cvtColor(np.expand_dims(img_pixels, axis=0), cv2.COLOR_BGR2HSV).squeeze()
    val = hsv_partial[:, 2]

    # compute gamma = log(mid*255)/log(mean)
    mid = 0.75
    mean = np.mean(val)
    gamma = math.log(mid*255)/math.log(mean)

    hue,sat,val = cv2.split(hsv)
    # do gamma correction on value channel
    val_gamma = np.power(val, gamma).clip(0,255).astype(np.uint8)

    # combine new value channel with original hue and sat channels
    hsv_gamma = cv2.merge([hue, sat, val_gamma])
    img_gamma2 = cv2.cvtColor(hsv_gamma, cv2.COLOR_HSV2BGR)
    return img_gamma2

def load_wb_model():
    net_awb = deep_wb_single_task.deepWBnet()
    net_awb.to(device='cpu')
    net_awb.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), 'models/net_awb.pth'), map_location='cpu'))
    net_awb.eval()
    return net_awb

def auto_white_balancing(img, net_awb):
    out_awb = deep_wb(img, task="awb", net_awb=net_awb, device="cpu", s=656)
    result_awb = np.array(out_awb, dtype=np.float32)*255
    result_awb = result_awb.astype(np.uint8)

    return result_awb


