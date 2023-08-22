import torch
import numpy as np
import math
import os
import sys
import tensorflow as tf
import cv2
import argparse

def load_intrinsic(intrin):
    # read intrinsics
    cam = np.zeros((2, 4, 4))
    intrinsics = open(intrin, 'r')
    words = intrinsics.read().split()
    intrinsics.close()
    words = [float(i) for i in words]
    words = np.asarray(words)
    words.reshape(4, 4)
    for i in range(0, 4):
        for j in range(0, 4):
            index = 4 * i + j
            cam[1][i][j] = words[index]
    cam[1][3][1] = cam[1][3][1] * 1.6
    return cam


def load_extrinsic(extrin, cam):
    extrinsics = open(extrin, 'r')
    words = extrinsics.read().split()
    extrinsics.close()
    words = [float(i) for i in words]
    words = np.asarray(words)
    words.reshape(4, 4)
    for i in range(0, 4):
        for j in range(0, 4):
            index = 4 * i + j
            cam[0][i][j] = words[index]
    return cam


def center_image(img):
    """ normalize image input """
    img = img.astype(np.float32)
    var = np.var(img, axis=(0,1), keepdims=True)
    mean = np.mean(img, axis=(0,1), keepdims=True)
    return (img - mean) / (np.sqrt(var) + 0.00000001)


def _pixel2cam(depth, pixel_coords, intrinsic_mat_inv):
  """Transform coordinates in the pixel frame to the camera frame."""
  cam_coords = tf.matmul(intrinsic_mat_inv, pixel_coords) * depth
  return cam_coords


def _meshgrid_abs(height, width):
  """Meshgrid in the absolute coordinates."""
  x_t = tf.matmul(
      tf.ones(shape=tf.stack([height, 1])),
      tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
  y_t = tf.matmul(
      tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
      tf.ones(shape=tf.stack([1, width])))
  x_t = (x_t + 1.0) * 0.5 * tf.cast(width - 1, tf.float32)
  y_t = (y_t + 1.0) * 0.5 * tf.cast(height - 1, tf.float32)
  x_t_flat = tf.reshape(x_t, (1, -1))
  y_t_flat = tf.reshape(y_t, (1, -1))
  ones = tf.ones_like(x_t_flat)
  grid = tf.concat([x_t_flat, y_t_flat, ones], axis=0)
  return grid


def inverse_warping(img,left_cam, right_cam, depth):
    # cameras (K, R, t)
    R_left = tf.slice(left_cam, [0, 0, 0, 0], [-1, 1, 3, 3])
    R_right = tf.slice(right_cam, [0, 0, 0, 0], [-1, 1, 3, 3])
    t_left = tf.slice(left_cam, [0, 0, 0, 3], [-1, 1, 3, 1])
    t_right = tf.slice(right_cam, [0, 0, 0, 3], [-1, 1, 3, 1])
    K_left = tf.slice(left_cam, [0, 1, 0, 0], [-1, 1, 3, 3])
    K_right = tf.slice(right_cam, [0, 1, 0, 0], [-1, 1, 3, 3])

    K_left = tf.squeeze(K_left, axis=1)

    K_left_inv = tf.linalg.inv(K_left)
    R_left_trans = tf.transpose(tf.squeeze(R_left, axis=1), perm=[0, 2, 1])
    R_right_trans = tf.transpose(tf.squeeze(R_right, axis=1), perm=[0, 2, 1])

    R_left = tf.squeeze(R_left, axis=1)
    t_left = tf.squeeze(t_left, axis=1)
    R_right = tf.squeeze(R_right, axis=1)
    t_right = tf.squeeze(t_right, axis=1)

    ## estimate egomotion by inverse composing R1,R2 and t1,t2
    R_rel = tf.matmul(R_right, R_left_trans)
    t_rel = tf.subtract(t_right, tf.matmul(R_rel, t_left))
    ## now convert R and t to transform mat, as in SFMlearner
    batch_size = R_left.shape[0]
    filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
    filler = tf.tile(filler, [batch_size, 1, 1])
    transform_mat = tf.concat([R_rel, t_rel], axis=2)
    transform_mat = tf.concat([transform_mat, filler], axis=1)

    dims = tf.shape(img)
    batch_size, img_height, img_width = dims[0], dims[1], dims[2]
    depth = tf.reshape(depth, [batch_size, 1, img_height * img_width])
    grid = _meshgrid_abs(img_height, img_width)
    grid = tf.tile(tf.expand_dims(grid, 0), [batch_size, 1, 1])
    cam_coords = _pixel2cam(depth, grid, K_left_inv)
    ones = tf.ones([batch_size, 1, img_height * img_width])
    cam_coords_hom = tf.concat([cam_coords, ones], axis=1)

    # Get projection matrix for target camera frame to source pixel frame
    hom_filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
    hom_filler = tf.tile(hom_filler, [batch_size, 1, 1])

    intrinsic_mat_hom = tf.concat(
        [K_left, tf.zeros([batch_size, 3, 1])], axis=2)
    intrinsic_mat_hom = tf.concat([intrinsic_mat_hom, hom_filler], axis=1)
    proj_target_cam_to_source_pixel = tf.matmul(intrinsic_mat_hom, transform_mat)
    source_pixel_coords = _cam2pixel(cam_coords_hom,
                                     proj_target_cam_to_source_pixel)
    source_pixel_coords = tf.reshape(source_pixel_coords,
                                     [batch_size, 2, img_height, img_width])
    source_pixel_coords = tf.transpose(source_pixel_coords, perm=[0, 2, 3, 1])
    warped_right, mask = _spatial_transformer(img, source_pixel_coords)
    return warped_right,mask


def _cam2pixel(cam_coords, proj_c2p):
  """Transform coordinates in the camera frame to the pixel frame."""
  pcoords = tf.matmul(proj_c2p, cam_coords)
  x = tf.slice(pcoords, [0, 0, 0], [-1, 1, -1])
  y = tf.slice(pcoords, [0, 1, 0], [-1, 1, -1])
  z = tf.slice(pcoords, [0, 2, 0], [-1, 1, -1])
  x_norm = x / (z + 1e-10)
  y_norm = y / (z + 1e-10)
  pixel_coords = tf.concat([x_norm, y_norm], axis=1)
  return pixel_coords


def _spatial_transformer(img, coords):
  """A wrapper over binlinear_sampler(), taking absolute coords as input."""
  img_height = tf.cast(tf.shape(img)[1], tf.float32)
  img_width = tf.cast(tf.shape(img)[2], tf.float32)
  px = coords[:, :, :, :1]
  py = coords[:, :, :, 1:]
  # Normalize coordinates to [-1, 1] to send to _bilinear_sampler.
  px = px / (img_width - 1) * 2.0 - 1.0
  py = py / (img_height - 1) * 2.0 - 1.0
  output_img, mask = _bilinear_sampler(img, px, py)
  return output_img, mask



def _bilinear_sampler(im, x, y, name='blinear_sampler'):
  x = tf.reshape(x, [-1])
  y = tf.reshape(y, [-1])
  batch_size = tf.shape(im)[0]
  _, height, width, channels = im.get_shape().as_list()
  x = tf.cast(x, 'float32')
  y = tf.cast(y, 'float32')
  height_f = tf.cast(height, 'float32')
  width_f = tf.cast(width, 'float32')
  zero = tf.constant(0, dtype=tf.int32)
  max_y = tf.cast(tf.shape(im)[1] - 1, 'int32')
  max_x = tf.cast(tf.shape(im)[2] - 1, 'int32')

  # Scale indices from [-1, 1] to [0, width - 1] or [0, height - 1].
  x = (x + 1.0) * (width_f - 1.0) / 2.0
  y = (y + 1.0) * (height_f - 1.0) / 2.0

  # Compute the coordinates of the 4 pixels to sample from.
  x0 = tf.cast(tf.floor(x), 'int32')
  x1 = x0 + 1
  y0 = tf.cast(tf.floor(y), 'int32')
  y1 = y0 + 1

  mask = tf.logical_and(
  tf.logical_and(x0 >= zero, x1 <= max_x),
  tf.logical_and(y0 >= zero, y1 <= max_y))
  mask = tf.cast(mask, 'float32')

  x0 = tf.clip_by_value(x0, zero, max_x)
  x1 = tf.clip_by_value(x1, zero, max_x)
  y0 = tf.clip_by_value(y0, zero, max_y)
  y1 = tf.clip_by_value(y1, zero, max_y)
  dim2 = width
  dim1 = width * height

  # Create base index.
  base = tf.range(batch_size) * dim1
  base = tf.reshape(base, [-1, 1])
  base = tf.tile(base, [1, height * width])
  base = tf.reshape(base, [-1])

  base_y0 = base + y0 * dim2
  base_y1 = base + y1 * dim2
  idx_a = base_y0 + x0
  idx_b = base_y1 + x0
  idx_c = base_y0 + x1
  idx_d = base_y1 + x1

  # Use indices to lookup pixels in the flat image and restore channels dim.
  im_flat = tf.reshape(im, tf.stack([-1, channels]))
  im_flat = tf.cast(im_flat, 'float32')
  pixel_a = tf.gather(im_flat, idx_a)
  pixel_b = tf.gather(im_flat, idx_b)
  pixel_c = tf.gather(im_flat, idx_c)
  pixel_d = tf.gather(im_flat, idx_d)

  x1_f = tf.cast(x1, 'float32')
  y1_f = tf.cast(y1, 'float32')

  # And finally calculate interpolated values.
  wa = tf.expand_dims(((x1_f - x) * (y1_f - y)), 1)
  wb = tf.expand_dims((x1_f - x) * (1.0 - (y1_f - y)), 1)
  wc = tf.expand_dims(((1.0 - (x1_f - x)) * (y1_f - y)), 1)
  wd = tf.expand_dims(((1.0 - (x1_f - x)) * (1.0 - (y1_f - y))), 1)

  output = tf.add_n([wa * pixel_a, wb * pixel_b, wc * pixel_c, wd * pixel_d])
  output = tf.reshape(output, tf.stack([batch_size, height, width, channels]))
  mask = tf.reshape(mask, tf.stack([batch_size, height, width, 1]))
  return output, mask

def resize(img):
    return cv2.resize(img, (640,480))

def calculate_photo_loss(args):
    images = []
    cams = []
    reference_image = cv2.imread(args.ref_rgb, 1)
    reference_image = resize(reference_image)
    images.append(reference_image)
    list_neigh = os.listdir(args.neighbours)
    for i in range(args.num_neighbours):
        neighbour = cv2.imread(list_neigh[i], 1)
        neighbour = resize(neighbour)
        images.append(neighbour)

    depth_image = cv2.imread(args.ref_depth, 0)
    depth_image = resize(depth_image)

    cam_intrin = load_intrinsic(args.intrinsics)

    ref_pose = load_extrinsic(args.ref_pose, cam_intrin)

    neigh_pose = os.listdir(args.neigh_pose)
    cam0 = load_extrinsic(neigh_pose[0], cam_intrin)
    cams.append(cam0)
    cams.append(ref_pose)
    for i in range(1, len(neigh_pose)):
        cam = load_extrinsic(neigh_pose[i], cam_intrin)
        cams.append(cam)

    dmap = torch.FloatTensor(torch.from_numpy(depth_image.astype(np.float32)))
    images = torch.FloatTensor(torch.from_numpy(np.stack(images, axis=0).astype(np.float32)))[None]
    cams = torch.FloatTensor(torch.from_numpy(np.stack(cams, axis=0).astype(np.float32)))[None]
    warped = []
    ref_cam = tf.squeeze(tf.slice(cams, [0, 0, 0, 0, 0], [1, 1, 2, 4, 4]), axis=1)
    for view in range(1, args.num_neighbours):
        view_cam = tf.squeeze(tf.slice(cams, [0, view, 0, 0, 0], [1, 1, 2, 4, 4]), axis=1)
        view_img = tf.slice(torch.squeeze(images), [view, 0, 0, 0], [1, -1, -1, 3])
        warped_view, mask = inverse_warping(view_img, ref_cam, view_cam, dmap)
        warped.append(warped_view)
    warped_view = torch.from_numpy(np.squeeze(np.array(warped)))
    warped_view = torch.mean(warped_view, -1)
    cost_volume, _ = torch.min(warped_view, dim=0, keepdim=True)
    loss = torch.mean(cost_volume)
    print(loss)
    return loss

parser = argparse.ArgumentParser(description='Robust Photometric Loss Calculation')
parser.add_argument('--ref_rgb', help='The reference view RGB Image', required=True)
parser.add_argument('--ref_depth', help='The reference image depth map', required=True)
parser.add_argument('--num_neighbours', help='Number of neighbouring viewpoints for warping', required=True)
parser.add_argument('--neighbours', help='Path to neighbours RGB images', required=True)
parser.add_argument('--intrinsics', help='Path to .txt intrinsics file', required=True)
parser.add_argument('--ref_pose', help='Path to pose .txt of reference image file', required=True)
parser.add_argument('--neigh_pose', help='Path to .txt pose folder of neighbour images', required=True)

args = parser.parse_args()
calculate_photo_loss(args)
