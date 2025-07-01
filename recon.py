import random
import shutil
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
import numpy as np
from tqdm import tqdm
from imageio.v3 import imread, imwrite
import os
import glob
import math
import nvdiffrast.torch as dr
import cv2
import trimesh
import copy
import argparse

from utils.nv_utils import *
from utils.dc_utils import read_images_frames, read_video_frames
from utils.render_utils import get_rays_from_pose
import time

import sys
sys.path.append("DepthCrafter")

from utils.depth_utils import DepthCrafterDemo

def point_to_mesh_cuda(pts, rgbs, faces, old_depth_src, min_angle_deg=2.5, target_normal=None, min_cos_sim=0.5, render_type="left",
    filter_type="angle"):
    h, w = rgbs.shape[:2]
    vertices = pts.reshape(-1, 3)
    # colors = rgbs.reshape(-1, 3)
    masks = torch.ones((h, w, 1), dtype=torch.uint8).to(rgbs.device) * 255
    rgbs = torch.cat([rgbs, masks], axis=-1)
    colors = rgbs.reshape(-1, 4)
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    face_normals = torch.cross(v1 - v0, v2 - v0)
    face_normals = F.normalize(face_normals, dim=1)
    if filter_type == "angle":
        # print(v0.shape, v1.shape, v2.shape)
        def angle_between(v1, v2):
            cos_theta = torch.sum(v1 * v2, -1) / (
                torch.norm(v1, dim=1) * torch.norm(v2, dim=1) + 1e-12
            )
            return torch.arccos(torch.clip(cos_theta, -1.0, 1.0)) * 180 / np.pi

        a = angle_between(v1 - v0, v2 - v0)
        b = angle_between(v2 - v1, v0 - v1)
        c = angle_between(v0 - v2, v1 - v2)
        min_angles = torch.minimum(torch.minimum(a, b), c)

        # Filter faces based on minimum angle
        valid_faces = min_angles >= min_angle_deg
        z_range = vertices[:, 2].max() - vertices[:, 2].min()

        z01, z12, z20 = torch.abs((v0 - v1)[:, 2]), torch.abs((v1 - v2)[:, 2]), torch.abs((v2 - v0)[:, 2])
        y01, y12, y20 = torch.abs((v0 - v1)[:, 1]), torch.abs((v1 - v2)[:, 1]), torch.abs((v2 - v0)[:, 1])
        x01, x12, x20 = torch.abs((v0 - v1)[:, 0]), torch.abs((v1 - v2)[:, 0]), torch.abs((v2 - v0)[:, 0])
        z_max = torch.maximum(torch.maximum(z01, z12), z20)
        y_max = torch.maximum(torch.maximum(y01, y12), y20)
        x_max = torch.maximum(torch.maximum(x01, x12), x20)
        proj_max = torch.maximum(torch.maximum(x_max, y_max), z_max)
        valid_faces2 = (proj_max / z_range < 0.013)
        valid_faces_final = valid_faces & valid_faces2
        if args.use_dwmesh:
            invalid_faces = faces[~valid_faces]
            colors.index_put_((invalid_faces,), torch.zeros(4).to(colors.device))
        else:
            faces = faces[valid_faces]
    elif filter_type == "boundary":
        z01, z12, z20 = torch.abs((v0 - v1)[:, 2]), torch.abs((v1 - v2)[:, 2]), torch.abs((v2 - v0)[:, 2])
        y01, y12, y20 = torch.abs((v0 - v1)[:, 1]), torch.abs((v1 - v2)[:, 1]), torch.abs((v2 - v0)[:, 1])
        x01, x12, x20 = torch.abs((v0 - v1)[:, 0]), torch.abs((v1 - v2)[:, 0]), torch.abs((v2 - v0)[:, 0])
        z_max = torch.maximum(torch.maximum(z01, z12), z20)
        y_max = torch.maximum(torch.maximum(y01, y12), y20)
        x_max = torch.maximum(torch.maximum(x01, x12), x20)
        threshold = torch.maximum(x_max, y_max)
        invalid = z_max / (threshold + 1e-8) > 2
        invalid_faces = faces[invalid]
        colors.index_put_((invalid_faces,), torch.zeros(4).to(colors.device))
    return vertices, faces, colors, None


def render_nvdiffrast(glctx, vertices, faces, colors, proj, poses, fovx, fovy, h, w, near=1e-3, far=1e3):
    # x right y down z forward
    def transform_pos(mtx, pos):
        t_mtx = torch.from_numpy(mtx).to(pos.device) if isinstance(mtx, np.ndarray) else mtx
        # (x,y,z) -> (x,y,z,1)
        posw = torch.cat([pos, torch.ones([pos.shape[0], 1]).to(pos.device)], axis=1)
        return torch.matmul(posw, t_mtx.t())[None, ...]

    def render(glctx, mtx, pos, pos_idx, vtx_col, col_idx, h, w):
        pos_clip    = transform_pos(mtx, pos)
        rast_out, _ = dr.rasterize(glctx, pos_clip, pos_idx, resolution=[h, w])
        visible_faces = rast_out[...,3][rast_out[...,3] > 0] - 1
        color, _    = dr.interpolate(vtx_col[None, ...], rast_out, col_idx)
        color       = dr.antialias(color, rast_out, pos_clip, pos_idx)
        return color
    poses[0,:] *= -1
    poses[1,:] *= -1
    poses[2,:] *= -1
    mvp = proj @ poses
    return render(glctx, mvp, vertices, faces, colors, faces, h, w)

def get_camera_pose(eye, center):
    up = np.array((0, 1, 0), dtype=np.float32)
    def normalize(v):
        norm = np.linalg.norm(v)
        if norm < 1e-8:
            return v
        return v / norm
    forward = normalize(center - eye)  # Camera looking direction
    right = normalize(np.cross(forward, up))  # Right vector (note: forward cross up for right-handed system)
    new_up = normalize(np.cross(right, forward))  # Recalculate up to ensure orthogonality
    view = np.zeros((4, 4), dtype=np.float32)
    view[0, 0:3] = right
    view[1, 0:3] = new_up
    view[2, 0:3] = forward
    view[0:3, 3] = -np.array([np.dot(right, eye), np.dot(new_up, eye), np.dot(forward, eye)])
    view[3, 3] = 1.0
    return torch.from_numpy(view)

def random_camera_traj(n_frames, depth_src, random_type, args, scene_name, depth_min=None, compare=True, fixed_frame=49, random_iter=0):
    rounds = (fixed_frame / args.num_frames) if fixed_frame > args.num_frames else args.num_frames / fixed_frame
    np.random.seed((hash(scene_name) + random_iter) % (2^32-1))
    if random_type == "180":
        radius = depth_min
        eyes = np.zeros((args.num_frames, 3))
        angle = np.linspace(0, 2 * (rounds) * np.pi, args.num_frames)
        eyes[:, 0] = np.sin(angle) * radius
        eyes[:, 1] = (np.cos(angle) - 1) * radius * 0.2

        centers = np.zeros((args.num_frames, 3))
        centers[:, 2] = radius
    else:
        ta = int(random_type)
        angle = np.linspace(0, (ta / 180 * np.pi) * rounds, args.num_frames)
        eyes = np.zeros((args.num_frames, 3))
        radius = depth_min
        eyes[:, 0] = np.sin(angle) * radius
        eyes[:, 1] = (np.abs(np.cos(angle)) - 1) * radius * 0.2
        eyes[:, 2] = radius - radius * np.abs(np.cos(angle))
        centers = np.zeros((args.num_frames, 3))
        centers[:, -1] = radius
    camera_poses = torch.stack([get_camera_pose(eye, center) for eye, center in zip(eyes, centers)], 0)
    camera_poses = camera_poses.to(depth_src.device)
    return camera_poses

def generate_faces(H, W, C, idx, device, left_padding=0, right_padding=0, top_padding=0, bottom_padding=0):
    idx = np.arange(H * W).reshape(H, W)
    idx = idx[top_padding:H-bottom_padding, left_padding:W-right_padding]
    faces = torch.from_numpy(np.concatenate([
            np.stack([idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[:-1, 1:].ravel()], axis=-1),
            np.stack([idx[:-1, 1:].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel()], axis=-1)
        ], axis=0)).int().to(device)
    faces = faces[:,[1,0,2]]
    return faces

def run_depth(frames, video_depth_anything, args, target_fps, DEVICE):
    height, width = frames.shape[1:3]
    depths, fps = video_depth_anything.infer_video_depth(frames, target_fps, input_size=args.input_size, device=DEVICE)
    depths = torch.from_numpy(depths).float().to(DEVICE)
    depths = F.interpolate(depths.unsqueeze(1), (height, width), mode='bilinear', align_corners=True).squeeze(1)
    # for more precise intrinsic, use MoGe.
    f = 500
    cx = depths.shape[-1] // 2
    cy = depths.shape[-2] // 2
    intrinsics = torch.tensor([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])
    return depths, intrinsics

def run_depth_crafter(frames, depth_estimater, near=0.0001, far=10000, 
    depth_inference_steps=5, depth_guidance_scale=1.0, window_size=110, overlap=25):
    depths, _ = depth_estimater.infer(
        frames,
        near,
        far,
        depth_inference_steps,
        depth_guidance_scale,
        window_size=window_size,
        overlap=overlap
    )
    f = 500
    cx = depths.shape[-1] // 2
    cy = depths.shape[-2] // 2
    intrinsics = torch.tensor([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])
    return depths, intrinsics

def run_render(video_path, output_dir, args, models):
    device = "cuda:0"
    print(f"Processing {video_path}")
    # 1. load video or images
    # frames = imread(video_path)
    frames, _ = read_video_frames(video_path, process_length=args.num_frames, max_res=1024)
    if frames.shape[0] < args.num_frames:
        print(f"Video {video_path} has less than {args.num_frames} frames, skipping")
        return
    scene_name = os.path.basename(video_path).split(".")[0]

    glctx = dr.RasterizeCudaContext(device=device)
    n_frames = min(args.num_frames, frames.shape[0])
    frames = frames[:n_frames]

    # 1.1 save gt
    if args.save_gt:
        os.makedirs(os.path.join(output_dir, scene_name), exist_ok=True)
        imwrite(os.path.join(output_dir, scene_name, "gt.mp4"), frames.astype(np.uint8), fps=30)
    
    depth_src, intrinsics = run_depth_crafter(frames.astype(np.float32) / 255., models)
    frames = torch.from_numpy(frames).float().to(device)

    depth_src = depth_src.to(device)

    # 3. render mesh
    with torch.no_grad():
        old_depth_src = depth_src
        depth_src = depth_src.clone()
        if args.debug == 0:
            if len(depth_src.shape) == 3:
                depth_src[:, 0, :] = 100
                depth_src[:, -1, :] = 100
                depth_src[:, 1:-1, 0] = 100
                depth_src[:, 1:-1, -1] = 100
            else:
                depth_src[:,:, 0, :] = 100
                depth_src[:,:, -1, :] = 100
                depth_src[:,:, 1:-1, 0] = 100
                depth_src[:,:, 1:-1, -1] = 100
        depth_src = depth_src.unsqueeze(-1)
        rgbs_src = frames

        H, W, C = rgbs_src[0].shape
        fidx = np.arange(H * W).reshape(H, W)
        fov_y = 2 * math.atan2(H, 2 * intrinsics[1, 1])
        fov_x = 2 * math.atan2(W, 2 * intrinsics[0, 0])
        # Create a camera intrinsic matrix, convert fov_deg to focal length
        fx = fy = 0.5 * H / math.tan(fov_y / 2)
        K = torch.tensor([[fx, 0, W / 2],
                            [0, fy, H / 2],
                            [0, 0, 1]], dtype=torch.float32, device=device)
        pose = torch.eye(4, device=device)
        ro_src, rd_src = get_rays_from_pose(pose, K, H, W)  # (H, W, 3)  
        proj = getprojection(fov_x, fov_y, n=1e-3, f=1e3, device=device)
        # random camera traj & rendering
        depth_min = depth_src[0].min().item() + 0.15
        camera_poses = random_camera_traj(n_frames=n_frames, depth_src=depth_src, random_type=args.cam,
            args=args, scene_name=scene_name, depth_min=depth_min)
        video = []
        random_type = args.cam
        for idx, poses in enumerate(tqdm(camera_poses)):
            if args.only_first_frame == 0 or idx == 0:
                pts_color = rgbs_src[idx]                    
                pts_xyz = depth_src[idx] * rd_src + ro_src  # (H, W, 3)
                faces = generate_faces(H, W, C, fidx, device)
                vertices, new_faces, colors, _ = point_to_mesh_cuda(pts_xyz, pts_color, faces, old_depth_src[idx], render_type=args.cam, filter_type=args.filter_type)
                if idx == 0 and (args.debug or args.save_mesh):
                    mesh = trimesh.Trimesh(vertices.cpu().numpy(), new_faces.cpu().numpy())
                    # Convert RGBA colors to RGBA uint8 format that trimesh expects
                    vertex_colors = colors.cpu().numpy().astype(np.uint8)[:, :3]
                    mesh.visual.vertex_colors = vertex_colors
                    mesh_path = os.path.join(output_dir, scene_name, "mesh.ply")
                    mesh.export(mesh_path)
            img = render_nvdiffrast(glctx, vertices, new_faces, colors, proj, poses, fov_x, fov_y, H, W)[0]
            if idx == 0:
                img[..., :3] = pts_color
                img[..., 3:] = 255
            else:
                mask = img[..., 3:]
                mask[mask > 125] = 255
                mask[mask <= 125] = 0
                img[..., 3:] = mask
                img[..., :3] = img[..., :3] * (mask / 255)
            video.append(img.cpu().numpy().astype(np.uint8))
        
        # output_dir = args.output_dir
        os.makedirs(os.path.join(output_dir, scene_name), exist_ok=True)

        cond_path = os.path.join(output_dir, scene_name, f"render_{random_type}.mp4")
        video = np.stack(video, axis=0).astype(np.uint8)
        # save to video via imageio.v3
        imwrite(cond_path, video[..., :3], fps=30)

        mask_path = os.path.join(output_dir, scene_name, f"mask_{random_type}.mp4")
        imwrite(mask_path, video[..., 3:], fps=30)
        if args.save_camera:
            # camera_poses = torch.stack(camera_poses, 0)
            output_video_path = os.path.join(output_dir, scene_name, f"camera_{random_type}.npz")
            np.savez_compressed(output_video_path, extrinsics=camera_poses.cpu().numpy(), intrinsics=K.cpu().numpy())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Video Depth Anything')
    parser.add_argument('--input_video', type=str, default="testset/origin_data/openvid")
    parser.add_argument('--num_frames',type=int,default=49)
    parser.add_argument('--output_dir', type=str, default="testset/processed/openvid")
    parser.add_argument('--debug',type=int,default=0)
    parser.add_argument('--only_first_frame',type=int,default=0)
    parser.add_argument('--save_depth',type=int,default=1)
    parser.add_argument('--cam',type=str,default="180")
    parser.add_argument('--save_camera',type=int,default=1)
    parser.add_argument('--save_mesh',type=int,default=0)
    parser.add_argument('--save_gt',type=int,default=0)
    parser.add_argument('--filter_type',type=str,default="angle")
    parser.add_argument('--use_dwmesh',type=int,default=1)
    parser.add_argument('--input_size', type=int, default=518)
    args = parser.parse_args()

    device = "cuda:0"
    depth_estimater = DepthCrafterDemo(
        unet_path="/mnt/bn/pico-mr-hl-taohu/Data/codes/checkpoints/DepthCrafter",
        pre_train_path="/mnt/bn/pico-mr-hl-taohu/Data/codes/checkpoints/stable-video-diffusion-img2vid",
        cpu_offload=None,
        device=device
    )

    run_render(args.input_video, args.output_dir, args, depth_estimater)