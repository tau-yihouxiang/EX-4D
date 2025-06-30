import argparse
import os
import torch
import torchvision
from diffsynth import ModelManager, WanVideoPipeline, save_video, load_state_dict
from PIL import Image
from peft import LoraConfig, inject_adapter_in_model
import numpy as np
from diffsynth.models.camera import CamVidEncoder
import cv2


def load_video_frames(video_path, num_frames, height, width):
    """Load and process video frames"""
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if frame_count < num_frames:
        print(f"Warning: Video has only {frame_count} frames, but {num_frames} required")
    
    # Get frame indices
    if frame_count >= num_frames:
        indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)
    else:
        indices = np.arange(frame_count)
    
    frames = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (width, height))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    
    cap.release()
    
    # Pad or trim frames to match num_frames
    if len(frames) < num_frames:
        frames += [frames[-1]] * (num_frames - len(frames))
    else:
        frames = frames[:num_frames]
    
    frames = np.array(frames)
    video_tensor = torch.from_numpy(frames).permute(3, 0, 1, 2).float() / 255.0
    return video_tensor

def load_mask_frames(mask_path, num_frames, height, width):
    """Load and process mask frames"""
    cap = cv2.VideoCapture(mask_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if frame_count < num_frames:
        print(f"Warning: Mask video has only {frame_count} frames, but {num_frames} required")
    
    # Get frame indices
    if frame_count >= num_frames:
        indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)
    else:
        indices = np.arange(frame_count)
    
    frames = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (width, height))
            # Apply random erosion like in original code
            import random
            kernel = random.randint(2, 8)
            frame = cv2.erode(frame, np.ones((kernel, kernel), np.uint8), iterations=1)
            frames.append(frame)
    
    cap.release()
    
    # Pad or trim frames to match num_frames
    if len(frames) < num_frames:
        frames += [frames[-1]] * (num_frames - len(frames))
    else:
        frames = frames[:num_frames]
    
    frames = np.array(frames)
    mask_tensor = torch.from_numpy(frames).permute(3, 0, 1, 2).float()
    mask_tensor = (mask_tensor / 255.0 > 0.5).float()  # Binarize mask
    return mask_tensor


def add_lora_to_model(model, lora_rank=16, lora_alpha=16.0, lora_target_modules="q,k,v,o,ffn.0,ffn.2", 
                     init_lora_weights="kaiming", pretrained_path=None, state_dict_converter=None):
    """Add LoRA to model"""
    if init_lora_weights == "kaiming":
        init_lora_weights = True
        
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        init_lora_weights=init_lora_weights,
        target_modules=lora_target_modules.split(","),
    )
    model = inject_adapter_in_model(lora_config, model)
    
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.to(torch.float32)
    
    if pretrained_path is not None:
        state_dict = load_state_dict(pretrained_path)
        if state_dict_converter is not None:
            state_dict = state_dict_converter(state_dict)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        all_keys = [i for i, _ in model.named_parameters()]
        num_updated_keys = len(all_keys) - len(missing_keys)
        num_unexpected_keys = len(unexpected_keys)
        print(f"LORA: {num_updated_keys} parameters loaded from {pretrained_path}. {num_unexpected_keys} unexpected.")


def main():
    parser = argparse.ArgumentParser(description='Video Inference with EX-4D')
    # parser.add_argument('--input_video', type=str, required=True, help='Path to input video')
    parser.add_argument('--color_video', type=str, help='Path to condition video (optional, defaults to input_video)')
    parser.add_argument('--mask_video', type=str, required=True, help='Path to mask video')
    parser.add_argument('--output_video', type=str, required=True, help='Path to output video')
    parser.add_argument('--height', type=int, default=512, help='Output height')
    parser.add_argument('--width', type=int, default=512, help='Output width')
    parser.add_argument('--num_frames', type=int, default=49, help='Number of frames to process')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--num_inference_steps', type=int, default=25, help='Number of inference steps')
    
    # Model paths
    parser.add_argument('--ex4d_path', type=str, default='models/EX-4D/ex4d.ckpt', help='Path to EX-4D model')
    parser.add_argument('--text_encoder_path', type=str, 
                       default='models/Wan-AI/Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth',
                       help='Path to text encoder')
    parser.add_argument('--vae_path', type=str,
                       default='models/Wan-AI/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth',
                       help='Path to VAE model')
    parser.add_argument('--clip_path', type=str,
                       default='models/Wan-AI/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth',
                       help='Path to CLIP model')
    parser.add_argument('--dit_dir', type=str,
                       default='models/Wan-AI/Wan2.1-I2V-14B-480P/',
                       help='Directory containing DiT model files')
    
    args = parser.parse_args()
    
    # Set condition video path
    color_video_path = args.color_video if args.color_video else args.input_video
    if not os.path.exists(color_video_path):
        raise FileNotFoundError(f"Condition video not found: {color_video_path}")
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output_video), exist_ok=True)
    
    # Load videos
    print(f"Loading mask video: {args.mask_video}")
    mask_tensor = load_mask_frames(args.mask_video, args.num_frames, args.height, args.width)
    
    print(f"Loading color video: {color_video_path}")
    color_tensor = load_video_frames(color_video_path, args.num_frames, args.height, args.width)

    color_tensor = (color_tensor * mask_tensor).to(torch.bfloat16) * 2 - 1  # Apply mask to color video
    mask_tensor = mask_tensor.to(torch.bfloat16) * 2 - 1  # Ensure mask is in bfloat16
    
    # Load models
    print("Loading models...")
    dit_paths = [
        os.path.join(args.dit_dir, f"diffusion_pytorch_model-{i:05d}-of-00007.safetensors")
        for i in range(1, 8)
    ]
    
    model_manager = ModelManager(torch_dtype=torch.bfloat16, device=args.device)
    model_manager.load_models([dit_paths, args.text_encoder_path, args.vae_path, args.clip_path])
    
    pipe = WanVideoPipeline.from_model_manager(model_manager, device=args.device)
    pipe.camera_encoder = CamVidEncoder(16, 1024, 5120).to(args.device, dtype=torch.bfloat16)
    
    # Add LoRA
    add_lora_to_model(pipe.denoising_model(), pretrained_path=args.ex4d_path, lora_rank=16, lora_alpha=16.0)
    pipe.load_cam(args.ex4d_path)
    pipe.enable_vram_management(num_persistent_param_in_dit=None)
    pipe.camera_encoder = pipe.camera_encoder.to(args.device, dtype=torch.bfloat16)
    pipe.camera_encoder.eval()
    
    # Set prompts
    prompt = "4K ultra HD, surround motion, realistic tone, panoramic shot, wide-angle view, cinematic quality"
    negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    
    # Run inference
    print("Running inference...")
    with torch.no_grad(), torch.amp.autocast(dtype=torch.bfloat16, device_type=args.device):
        input_cond = color_tensor.to(args.device)[None]  # Add batch dimension
        input_mask = mask_tensor.to(args.device)[None]
        
        output_video = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            video=input_cond,
            mask=input_mask,
            num_inference_steps=args.num_inference_steps,
            seed=args.seed,
            tiled=False,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
        )
    
    # Save output video
    print(f"Saving output video: {args.output_video}")
    save_video(output_video, args.output_video, fps=15, quality=8)
    print("Done!")


if __name__ == "__main__":
    main()