# Copyright (2025) Bytedance Ltd. and/or its affiliates 

# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 

#     http://www.apache.org/licenses/LICENSE-2.0 

# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License. 
import numpy as np
import torch
def compute_scale_and_shift(prediction, target, mask, scale_only=False):
    if scale_only:
        return compute_scale(prediction, target, mask), 0
    else:
        return compute_scale_and_shift_full(prediction, target, mask)


def compute_scale(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    prediction = prediction.astype(np.float32)
    target = target.astype(np.float32)
    mask = mask.astype(np.float32)

    a_00 = np.sum(mask * prediction * prediction)
    a_01 = np.sum(mask * prediction)
    a_11 = np.sum(mask)

    # right hand side: b = [b_0, b_1]
    b_0 = np.sum(mask * prediction * target)

    x_0 = b_0 / (a_00 + 1e-6)

    return x_0

# def compute_scale_and_shift_full(prediction, target, mask):
#     # system matrix: A = [[a_00, a_01], [a_10, a_11]]
#     prediction = prediction.astype(np.float32)
#     target = target.astype(np.float32)
#     mask = mask.astype(np.float32)

#     a_00 = np.sum(mask * prediction * prediction)
#     a_01 = np.sum(mask * prediction)
#     a_11 = np.sum(mask)

#     b_0 = np.sum(mask * prediction * target)
#     b_1 = np.sum(mask * target)

#     x_0 = 1
#     x_1 = 0

#     det = a_00 * a_11 - a_01 * a_01

#     if det != 0:
#         x_0 = (a_11 * b_0 - a_01 * b_1) / det
#         x_1 = (-a_01 * b_0 + a_00 * b_1) / det

#     return x_0, x_1

def compute_scale_and_shift_full(prediction, target, mask = None):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    prediction = prediction.float()
    target = target.float()
    if mask is None:
        mask = torch.ones_like(target)
    mask = mask.float()
    # print(mask.shape, prediction.shape, target.shape)
    a_00 = torch.sum(mask * prediction * prediction)
    a_01 = torch.sum(mask * prediction)
    a_11 = torch.sum(mask)

    b_0 = torch.sum(mask * prediction * target)
    b_1 = torch.sum(mask * target)

    x_0 = 1
    x_1 = 0

    det = a_00 * a_11 - a_01 * a_01

    if det != 0:
        x_0 = (a_11 * b_0 - a_01 * b_1) / det
        x_1 = (-a_01 * b_0 + a_00 * b_1) / det

    return x_0, x_1

def get_interpolate_frames(frame_list_pre, frame_list_post):
    assert len(frame_list_pre) == len(frame_list_post)
    min_w = 0.0
    max_w = 1.0
    step = (max_w - min_w) / (len(frame_list_pre)-1)
    post_w_list = [min_w] + [i * step for i in range(1,len(frame_list_pre)-1)] + [max_w]
    interpolated_frames = []
    for i in range(len(frame_list_pre)):
        interpolated_frames.append(frame_list_pre[i] * (1-post_w_list[i]) + frame_list_post[i] * post_w_list[i])
    return interpolated_frames

def convert_camera_to_trajcrafter(eyes, centers, traj_txt_path):
    # convert the data format to trajcrafter format
    # theta, phi, r, x, y
    theta = np.arctan2(centers[:,1] - eyes[:, 1], centers[0, 2]-eyes[0, 2])
    phi = np.arctan2(centers[:,0] - eyes[:, 0], centers[0, 2]-eyes[0, 2])
    r = (centers[:,2] - eyes[:, 2]) - (centers[0,2] - eyes[0, 2])
    x = (centers[:,0] - eyes[:, 0]) - (centers[0,0] - eyes[0, 0])
    y = (centers[:,1] - eyes[:, 1]) - (centers[0,1] - eyes[0, 1])
    # with open(traj_txt_path, 'w') as f:
    #     f.write(" ".join([str(t) for t in theta]).strip()+"\n")
    #     f.write(" ".join([str(t) for t in phi]).strip()+"\n")
    #     f.write(" ".join([str(t) for t in r]).strip()+"\n")
    #     f.write(" ".join([str(t) for t in x]).strip()+"\n")
    #     f.write(" ".join([str(t) for t in y]).strip()+"\n")
    # f.close()

def convert_camera_to_camctrl():
    pass

def convert_camera_to_recammaster():
    pass
