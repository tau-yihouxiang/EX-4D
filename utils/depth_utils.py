import gc
import os
import numpy as np
import torch
import cv2

from diffusers.training_utils import set_seed
import sys
sys.path.append("DepthCrafter")
from depthcrafter.depth_crafter_ppl import DepthCrafterPipeline
from depthcrafter.unet import DiffusersUNetSpatioTemporalConditionModelDepthCrafter

class DepthCrafterDemo:
    def __init__(
        self,
        unet_path: str,
        pre_train_path: str,
        cpu_offload: str = "model",
        device: str = "cuda:0",
    ):
        unet = DiffusersUNetSpatioTemporalConditionModelDepthCrafter.from_pretrained(
            unet_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )
        # load weights of other components from the provided checkpoint
        self.pipe = DepthCrafterPipeline.from_pretrained(
            pre_train_path,
            unet=unet,
            torch_dtype=torch.float16,
            variant="fp16",
        )

        # for saving memory, we can offload the model to CPU, or even run the model sequentially to save more memory
        if cpu_offload is not None:
            if cpu_offload == "sequential":
                # This will slow, but save more memory
                self.pipe.enable_sequential_cpu_offload()
            elif cpu_offload == "model":
                self.pipe.enable_model_cpu_offload()
            else:
                raise ValueError(f"Unknown cpu offload option: {cpu_offload}")
        else:
            self.pipe.to(device)
        # enable attention slicing and xformers memory efficient attention
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except Exception as e:
            print(e)
            print("Xformers is not enabled")
        self.pipe.enable_attention_slicing()

    def infer(
        self,
        frames,
        near,
        far,
        num_denoising_steps: int,
        guidance_scale: float,
        window_size: int = 110,
        overlap: int = 25,
        seed: int = 42,
        track_time: bool = True,
    ):
        set_seed(seed)

        # inference the depth map using the DepthCrafter pipeline
        with torch.inference_mode():
            print(self.pipe.device)
            res = self.pipe(
                frames,
                height=frames.shape[1],
                width=frames.shape[2],
                output_type="np",
                guidance_scale=guidance_scale,
                num_inference_steps=num_denoising_steps,
                window_size=window_size,
                overlap=overlap,
                track_time=track_time,
            ).frames[0]
        # convert the three-channel output to a single channel depth map
        res = res.sum(-1) / res.shape[-1]
        ori_depths = (res - res.min()) / (res.max() - res.min())
        # visualize the depth map and save the results
        # vis = vis_sequence_depth(res)
        # save the depth map and visualization with the target FPS
        depths = torch.from_numpy(ori_depths.copy()).unsqueeze(1)  # 49 576 1024 ->
        depths *= 3900  # compatible with da output
        depths[depths < 1e-5] = 1e-5
        depths = 10000.0 / depths
        depths = depths.clip(near, far)
        # print(ori_depths.mean(), ori_depths.min(), ori_depths.max())
        return depths, ori_depths

def extract_video(video_path):
    # create folder
    folder_name = os.path.basename(video_path).split(".")[0]
    os.makedirs(folder_name, exist_ok=True)
    os.makedirs(f"{folder_name}/depth", exist_ok=True)
    os.makedirs(f"{folder_name}/masks", exist_ok=True)
    os.makedirs(f"{folder_name}/images", exist_ok=True)
    # extract frames
    os.system(f"ffmpeg -i {video_path} {folder_name}/images/%06d.png")

def process_video_depth(video_path, method):
    if method == 'DepthPro':
        # Load model and preprocessing transform
        model, transform = depth_pro.create_model_and_transforms()
        model.eval()

        image_folder = os.path.basename(video_path).split(".")[0]
        # Load and preprocess an image.
        depths = []
        with torch.no_grad():
            for image_path in sorted(os.listdir(os.path.join(image_folder, "images"))):
                image, _, f_px = depth_pro.load_rgb(os.path.join(image_folder, "images", image_path))
                image = transform(image)
                # Run inference.
                prediction = model.infer(image, f_px=f_px)
                depth = prediction["depth"]  # Depth in [m].
                # focallength_px = prediction["focallength_px"]  # Focal length in pixels.
                print(depth.min(), depth.max(), depth.shape)
                # Save depth map.
                depth_path = os.path.join(image_folder, "depth", image_path.replace(".png", ".npy"))
                np.save(depth_path, depth.cpu().numpy())
    elif method == "DepthCrafter":
        depths = self.depth_estimater.infer(
            frames,
            0.001,
            10000,
            5,
            opts.depth_guidance_scale,
            window_size=1.0,
            overlap=25,
        ).to("cuda:0")
        # normalize the depth map to [0, 1] across the whole video
    else:
        raise NotImplementedError

def convert_npy_to_video(npy_dir, output_path):
    npy_files = sorted([f for f in os.listdir(npy_dir) if f.endswith('.npy')])
    
    # Read first file to get dimensions
    first_depth = np.load(os.path.join(npy_dir, npy_files[0]))
    height, width = first_depth.shape
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 10, (width, height), isColor=False)
    
    # Process each depth file
    for npy_file in npy_files:
        depth = np.load(os.path.join(npy_dir, npy_file))
        
        # Normalize depth to 0-255 range
        depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_uint8 = depth_normalized.astype(np.uint8)
        
        # Write frame to video
        out.write(depth_uint8)
    out.release()

if __name__ == "__main__":
    # extract_video("sea.mp4")
    # process_video_depth("sea.mp4", 'DepthPro')
    convert_npy_to_video("sea/depth", "sea_depth.mp4")