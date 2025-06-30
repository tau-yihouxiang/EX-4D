import torch
import numpy as np
import cv2
import imageio
import einops

def get_rays(directions, c2w):     # directions: (H*W, 3)
    # c2w: (4, 4)
    # return: rays_o (H*W, 3), rays_d (H*W, 3)
    
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = torch.einsum('kj,ij->ik', c2w[:3, :3], directions)  # (H*W, 3)
    
    # Normalize ray directions
    # rays_d = rays_d / (torch.norm(rays_d, dim=-1, keepdim=True) + 1e-12)
    
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:3, 3].unsqueeze(0).expand(directions.shape[0], -1)  # (H*W, 3)
    
    return rays_o, rays_d


def get_rays_from_pose(pose, K, H, W):
    # pose: (4, 4)
    # K: (3, 3)
    # H, W: int
    # return: (H, W, 3), (H, W, 3)
    
    # Create a meshgrid for screen coordinates
    rays_screen_coords = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    grid = torch.stack(rays_screen_coords, dim=-1).reshape(H * W, 2).to(pose)
    i, j = grid[..., 1], grid[..., 0]
    
    # Extract intrinsic parameters
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    
    # Calculate directions
    directions = torch.stack([
        (i - cx) / fx,
        (j - cy) / fy,
        torch.ones_like(i)
    ], dim=-1)  # (H*W, 3)
    
    # Get rays
    ro, rd = get_rays(directions, pose)  # (H*W, 3), (H*W, 3)
    
    # Reshape rays
    ro = ro.reshape(H, W, 3)
    rd = rd.reshape(H, W, 3)
    
    return ro, rd