import os
import cv2
import json
import numpy as np
import math
import numpy.matlib

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

def auto_white_balancing(img):
    wb = WBsRGB()
    img = wb.correctImage(img)
    return img

## White-balance model class
#
# Copyright (c) 2018-present, Mahmoud Afifi
# York University, Canada
# mafifi@eecs.yorku.ca | m.3afifi@gmail.com
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# All rights reserved.
#
# Please cite the following work if this program is used:
# Mahmoud Afifi, Brian Price, Scott Cohen, and Michael S. Brown,
# "When color constancy goes wrong: Correcting improperly white-balanced
# images", CVPR 2019.
#
##########################################################################

class WBsRGB:
  def __init__(self, gamut_mapping=1):

    self.features = np.load(os.path.join(os.path.dirname(__file__), 'models/features+.npy'))
    self.mappingFuncs = np.load(os.path.join(os.path.dirname(__file__), 'models/mappingFuncs+.npy'))
    self.encoderWeights = np.load(os.path.join(os.path.dirname(__file__), 'models/encoderWeights+.npy'))  # PCA weights
    self.encoderBias = np.load(os.path.join(os.path.dirname(__file__), 'models/encoderBias+.npy'))  # PCA bias
    self.K = 75

    self.sigma = 0.25  # fall-off factor for KNN blending
    self.h = 60  # histogram bin width
    # our results reported with gamut_mapping=2, however gamut_mapping=1
    # gives more compelling results with over-saturated examples.
    self.gamut_mapping = gamut_mapping  # options: 1 scaling, 2 clipping

  def encode(self, hist):
    """ Generates a compacted feature of a given RGB-uv histogram tensor."""
    histR_reshaped = np.reshape(np.transpose(hist[:, :, 0]),
                                (1, int(hist.size / 3)), order="F")
    histG_reshaped = np.reshape(np.transpose(hist[:, :, 1]),
                                (1, int(hist.size / 3)), order="F")
    histB_reshaped = np.reshape(np.transpose(hist[:, :, 2]),
                                (1, int(hist.size / 3)), order="F")
    hist_reshaped = np.append(histR_reshaped,
                              [histG_reshaped, histB_reshaped])
    feature = np.dot(hist_reshaped - self.encoderBias.transpose(),
                     self.encoderWeights)
    return feature

  def rgb_uv_hist(self, I):
    """ Computes an RGB-uv histogram tensor. """
    sz = np.shape(I)  # get size of current image
    if sz[0] * sz[1] > 202500:  # resize if it is larger than 450*450
      factor = np.sqrt(202500 / (sz[0] * sz[1]))  # rescale factor
      newH = int(np.floor(sz[0] * factor))
      newW = int(np.floor(sz[1] * factor))
      I = cv2.resize(I, (newW, newH), interpolation=cv2.INTER_NEAREST)
    I_reshaped = I[(I > 0).all(axis=2)]
    eps = 6.4 / self.h
    hist = np.zeros((self.h, self.h, 3))  # histogram will be stored here
    Iy = np.linalg.norm(I_reshaped, axis=1)  # intensity vector
    for i in range(3):  # for each histogram layer, do
      r = []  # excluded channels will be stored here
      for j in range(3):  # for each color channel do
        if j != i:
          r.append(j)
      Iu = np.log(I_reshaped[:, i] / I_reshaped[:, r[1]])
      Iv = np.log(I_reshaped[:, i] / I_reshaped[:, r[0]])
      hist[:, :, i], _, _ = np.histogram2d(
        Iu, Iv, bins=self.h, range=((-3.2 - eps / 2, 3.2 - eps / 2),) * 2, weights=Iy)
      norm_ = hist[:, :, i].sum()
      hist[:, :, i] = np.sqrt(hist[:, :, i] / norm_)  # (hist/norm)^(1/2)
    return hist

  def correctImage(self, I):
    """ White balance a given image I. """
    I = I[..., ::-1]  # convert from BGR to RGB
    I = im2double(I)  # convert to double
    # Convert I to float32 may speed up the process.
    feature = self.encode(self.rgb_uv_hist(I))
    # Do
    # ```python
    # feature_diff = self.features - feature
    # D_sq = np.einsum('ij,ij->i', feature_diff, feature_diff)[:, None]
    # ```
    D_sq = np.einsum(
      'ij, ij ->i', self.features, self.features)[:, None] + np.einsum(
      'ij, ij ->i', feature, feature) - 2 * self.features.dot(feature.T)

    # get smallest K distances
    idH = D_sq.argpartition(self.K, axis=0)[:self.K]
    mappingFuncs = np.squeeze(self.mappingFuncs[idH, :])
    dH = np.sqrt(
      np.take_along_axis(D_sq, idH, axis=0))
    weightsH = np.exp(-(np.power(dH, 2)) /
                      (2 * np.power(self.sigma, 2)))  # compute weights
    weightsH = weightsH / sum(weightsH)  # normalize blending weights
    mf = sum(np.matlib.repmat(weightsH, 1, 33) *
             mappingFuncs, 0)  # compute the mapping function
    mf = mf.reshape(11, 3, order="F")  # reshape it to be 9 * 3
    I_corr = self.colorCorrection(I, mf)  # apply it!
    return I_corr

  def colorCorrection(self, input, m):
    """ Applies a mapping function m to a given input image. """
    sz = np.shape(input)  # get size of input image
    I_reshaped = np.reshape(input, (int(input.size / 3), 3), order="F")
    kernel_out = kernelP(I_reshaped)
    out = np.dot(kernel_out, m)
    if self.gamut_mapping == 1:
      # scaling based on input image energy
      out = normScaling(I_reshaped, out)
    elif self.gamut_mapping == 2:
      # clip out-of-gamut pixels
      out = outOfGamutClipping(out)
    else:
      raise Exception('Wrong gamut_mapping value')
    # reshape output image back to the original image shape
    out = out.reshape(sz[0], sz[1], sz[2], order="F")
    out = out.astype('float32')[..., ::-1]  # convert from BGR to RGB
    return out


def normScaling(I, I_corr):
  """ Scales each pixel based on original image energy. """
  norm_I_corr = np.sqrt(np.sum(np.power(I_corr, 2), 1))
  inds = norm_I_corr != 0
  norm_I_corr = norm_I_corr[inds]
  norm_I = np.sqrt(np.sum(np.power(I[inds, :], 2), 1))
  I_corr[inds, :] = I_corr[inds, :] / np.tile(
    norm_I_corr[:, np.newaxis], 3) * np.tile(norm_I[:, np.newaxis], 3)
  return I_corr


def kernelP(rgb):
  """ Kernel function: kernel(r, g, b) -> (r,g,b,rg,rb,gb,r^2,g^2,b^2,rgb,1)
        Ref: Hong, et al., "A study of digital camera colorimetric
          characterization based on polynomial modeling." Color Research &
          Application, 2001. """
  r, g, b = np.split(rgb, 3, axis=1)
  return np.concatenate(
    [rgb, r * g, r * b, g * b, rgb ** 2, r * g * b, np.ones_like(r)], axis=1)


def outOfGamutClipping(I):
  """ Clips out-of-gamut pixels. """
  I[I > 1] = 1  # any pixel is higher than 1, clip it to 1
  I[I < 0] = 0  # any pixel is below 0, clip it to 0
  return I


def im2double(im):
  """ Returns a double image [0,1] of the uint8 im [0,255]. """
  return cv2.normalize(im.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)