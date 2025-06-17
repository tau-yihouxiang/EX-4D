# EX-4D: EXtreme Viewpoint 4D Video Synthesis via Depth Watertight Mesh

<div align="center">

<img src="docs/Logo.png" alt="EX-4D Logo" width="250">

[📄 Paper](https://arxiv.org/abs/2506.05554)  |  [🎥 Homepage](https://tau-yihouxiang.github.io/projects/EX-4D/EX-4D.html)  |  [💻 Code](https://github.com/tau-yihouxiang/EX-4D)

</div>



## 🌟 Highlights

- **🎯 Extreme Viewpoint Synthesis**: Generate high-quality 4D videos with camera movements ranging from -90° to 90°
- **🔧 Depth Watertight Mesh**: Novel geometric representation that models both visible and occluded regions
- **⚡ Lightweight Architecture**: Only 1% trainable parameters (140M) of the 14B video diffusion backbone
- **🎭 No Multi-view Training**: Innovative masking strategy eliminates the need for expensive multi-view datasets
- **🏆 State-of-the-art Performance**: Outperforms existing methods, especially on extreme camera angles

## 🎬 Demo Results

<div align="center">
<img src="docs/teaser.png" alt="EX-4D Demo Results" width="800">
</div>

*EX-4D transforms monocular videos into camera-controllable 4D experiences with physically consistent results under extreme viewpoints.*

## 🏗️ Framework Overview

<div align="center">
<img src="docs/overview.png" alt="EX-4D Architecture">
</div>

Our framework consists of three key components:

1. **🔺 Depth Watertight Mesh Construction**: Creates a robust geometric prior that explicitly models both visible and occluded regions
2. **🎭 Simulated Masking Strategy**: Generates effective training data from monocular videos without multi-view datasets
3. **⚙️ Lightweight LoRA Adapter**: Efficiently integrates geometric information with pre-trained video diffusion models

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/username/EX-4D.git
cd EX-4D

# Create conda environment
conda create -n ex4d python=3.10
conda activate ex4d

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```bash
pip install torch torchvision
pip install opencv-python
pip install nvdiffrast
pip install transformers
pip install diffusers
pip install accelerate
```

### Basic Usage

```python
import torch
from ex4d import EX4DModel

# Load pre-trained model
model = EX4DModel.from_pretrained("ex4d-base")

# Load input video
input_video = "path/to/your/video.mp4"

# Define camera trajectory (example: rotation from -90° to 90°)
camera_trajectory = generate_camera_trajectory(
    start_angle=-90, 
    end_angle=90, 
    num_frames=49
)

# Generate 4D video
output_video = model.generate(
    input_video=input_video,
    camera_trajectory=camera_trajectory,
    output_resolution=(512, 512)
)

# Save result
save_video(output_video, "output_4d_video.mp4")
```

### Training

```bash
# Download training dataset (OpenVID-1M)
python scripts/download_data.py --dataset openvid

# Start training
python train.py \
    --config configs/ex4d_base.yaml \
    --data_path data/openvid \
    --output_dir outputs/ex4d_training \
    --batch_size 4 \
    --learning_rate 3e-5 \
    --num_gpus 8
```

## 📊 Performance

### Quantitative Results

| Method | FID (Extreme) ↓ | FVD (Extreme) ↓ | VBench Score ↑ |
|--------|-----------------|-----------------|----------------|
| ReCamMaster | 64.68 | 943.45 | 0.434 |
| TrajectoryCrafter | 65.33 | 893.80 | 0.447 |
| TrajectoryAttention | 62.49 | 912.14 | 0.389 |
| **EX-4D (Ours)** | **55.42** | **823.61** | **0.450** |

### User Study Results

- **70.7%** of participants preferred EX-4D over baseline methods
- Superior performance in physical consistency and extreme viewpoint quality
- Significant improvement as camera angles become more extreme

## 🔧 Technical Details

### Depth Watertight Mesh Construction

```python
def construct_dw_mesh(depth_map, image):
    """
    Construct Depth Watertight Mesh from depth map and image
    
    Args:
        depth_map: Per-frame depth estimation [H, W]
        image: RGB image [H, W, 3]
    
    Returns:
        mesh: DW-Mesh {V, F, T, O}
    """
    # Unproject pixels to 3D vertices
    vertices = unproject_depth_to_3d(depth_map)
    
    # Create triangular faces
    faces = create_triangular_faces(vertices)
    
    # Detect occlusions and assign textures
    occlusions = detect_occlusions(faces, depth_map)
    textures = assign_textures(image, occlusions)
    
    return DWMesh(vertices, faces, textures, occlusions)
```

### LoRA Adapter Integration

```python
def add_lora_to_model(model, lora_rank=16, target_modules="q,k,v,o,ffn.0,ffn.2"):
    """
    Add LoRA adaptation to video diffusion model
    
    Args:
        model: Pre-trained video diffusion model
        lora_rank: Rank for low-rank adaptation
        target_modules: Target layers for LoRA injection
    """
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=16,
        target_modules=target_modules.split(","),
        init_lora_weights=True
    )
    return inject_adapter_in_model(lora_config, model)
```

## 📁 Project Structure

```
EX-4D/
├── configs/                 # Configuration files
│   ├── ex4d_base.yaml
│   └── ex4d_large.yaml
├── ex4d/                   # Main package
│   ├── models/             # Model implementations
│   │   ├── dw_mesh.py     # Depth Watertight Mesh
│   │   ├── adapter.py     # LoRA adapter
│   │   └── diffusion.py   # Video diffusion integration
│   ├── data/              # Data processing
│   └── utils/             # Utility functions
├── scripts/               # Training and evaluation scripts
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## 🎯 Applications

- **🎮 Gaming**: Create immersive 3D game cinematics from 2D footage
- **🎬 Film Production**: Generate novel camera angles for post-production
- **🥽 VR/AR**: Create free-viewpoint video experiences
- **📱 Social Media**: Generate dynamic camera movements for content creation
- **🏢 Architecture**: Visualize spaces from multiple viewpoints

## 📈 Benchmarks

### Viewpoint Range Evaluation

| Range | Small (0°→30°) | Large (0°→60°) | Extreme (0°→90°) | Full (-90°→90°) |
|-------|----------------|----------------|------------------|-----------------|
| FID Score | 44.19 | 50.30 | 55.42 | - |
| Performance Gap | +9.1% better | +8.9% better | +11.3% better | +15.5% better |

*Performance gap compared to the second-best method in each category.*

## ⚠️ Limitations

- **Depth Dependency**: Performance relies on monocular depth estimation quality
- **Fine Details**: May struggle with very thin structures or fine geometric details
- **Computational Cost**: Requires significant computation for high-resolution videos
- **Reflective Surfaces**: Challenges with reflective or transparent materials

## 🔮 Future Work

- [ ] Multi-frame depth consistency enforcement
- [ ] Uncertainty-aware mesh construction
- [ ] Neural mesh refinement techniques
- [ ] Real-time inference optimization
- [ ] Support for higher resolutions (1K, 2K)

## 📚 Citation

If you find our work useful, please consider citing:

```bibtex
@misc{hu2025ex4dextremeviewpoint4d,
      title={EX-4D: EXtreme Viewpoint 4D Video Synthesis via Depth Watertight Mesh}, 
      author={Tao Hu and Haoyang Peng and Xiao Liu and Yuewen Ma},
      year={2025},
      eprint={2506.05554},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.05554}, 
}
```
