from raycast_rgbd.raycast_rgbd import *
import numpy as np
import torch
import cv2
import argparse


def renderer(input_tsdf,outh, outw, voxsize,intrinsics, world2grid, pose):
    inputs = np.load(input_tsdf)
    inputs = inputs['arr_0']
    dim3d = inputs.shape
    inputs = np.rint(inputs)
    inputs = inputs[None][None]
    inputs = torch.from_numpy(inputs).cuda()

    input_locs = torch.nonzero(torch.abs(inputs[:, 0]) < 1)
    input_locs = torch.cat([input_locs[:, 1:], input_locs[:, :1]], 1)

    ray_increment = 0.3 * 1
    thresh_sample_dist = 50.5 * ray_increment
    raycast_depth_max = 6.0

    input_vals = inputs[input_locs[:, -1], :, input_locs[:, 0], input_locs[:, 1], input_locs[:, 2]]
    max_num_locs_per_sample = input_vals.shape[0]

    raycaster_rgbd = RaycastRGBD(1, dim3d, outw, outh, depth_min=0.1 / voxsize, depth_max=raycast_depth_max / voxsize,
                                 thresh_sample_dist=thresh_sample_dist, ray_increment=ray_increment,
                                 max_num_locs_per_sample=max_num_locs_per_sample)

    intrin = torch.tensor([[intrinsics[0][0], intrinsics[1][1],  intrinsics[0][2],  intrinsics[1][2]]])
    view_matrix = torch.matmul(world2grid, pose)

    normals = torch.zeros((max_num_locs_per_sample, 3))
    color = torch.zeros(normals.shape)
    raycast_color, depthrc, raycast_normal = raycaster_rgbd(input_locs.cuda(), input_vals[:, :1].contiguous(),
                                                            color.contiguous().cuda(), normals.cuda(),
                                                            view_matrix.cuda(),
                                                            intrinsics.cuda())
    depthrc = depthrc.permute(2, 1, 0).detach().cpu().data.numpy()
    print(depthrc.max())
    print(depthrc.min())
    return depthrc

def calc_world2grid(voxel_size,bounds, scenePad, heightPad):
    scale = np.full((4, 4), 1.0 / voxel_size)
    padVec = np.array([scenePad, scenePad, heightPad]).reshape(3, 1)
    padVec = padVec * voxel_size
    parameter_translation = -bounds.min() + padVec
    translation = np.identity(4)
    translation[0, 3], translation[1, 3], translation[2, 3] = parameter_translation[0, 0], parameter_translation[1, 0], \
                                                              parameter_translation[2, 0]
    world2grid = scale * translation
    world2grid[3, 3] = 1
    return world2grid


def load_param(path):
    intrinsics = open(path, 'r')
    words = intrinsics.read().split()
    intrinsics.close()
    words = [float(i) for i in words]
    words = np.asarray(words)
    words.reshape(4, 4)
    return words

def load_bounds(bounds):
    bound = open(bounds, 'r')
    words = bound.read().split()
    bound.close()
    words = [float(i) for i in words]
    words = np.asarray(words)
    bound.reshape(1,3)
    return bound

def main(args):
    intrin = load_param(args.intrinsics)
    pose = load_param(args.pose)
    bounds = load_bounds(args.bound)
    world2grid = calc_world2grid(args.voxelsize, bounds, scenePad=6,heightPad=3)
    depthrc = renderer(args.tsdf, args.outh, args.outw, args.voxelsize,intrin, world2grid, pose)
    cv2.imwrite(args.out_depth, depthrc)


parser = argparse.ArgumentParser(description='Differential Renderer')
parser.add_argument('--intrinsics', help='Path to txt intrinsics file', required=True)
parser.add_argument('--pose', help='Path to txt pose file', required=True)
parser.add_argument('--tsdf', help='Path to npz tsdf file with the 3d values labeled as tsdf array', required=True)
parser.add_argument('--out_depth', help='Path to the output depth image', required=True)
parser.add_argument('--outh', help='output depth image height', required=False, default=480)
parser.add_argument('--outw', help='output depth image width', required=False, default=640)
parser.add_argument('--voxelsize', help='the voxel size of the tsdf', required=False, default=0.046875)
parser.add_argument('--bounds', help='the .txt bounds of tsdf scene', required=True)
args = parser.parse_args()
main(args)