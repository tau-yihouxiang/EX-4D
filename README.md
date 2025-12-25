
<div align="right">
  <details>
    <summary >🌐 Language</summary>
    <div>
      <div align="right">
        <p><a href="https://openaitx.github.io/view.html?user=tau-yihouxiang&project=EX-4D&lang=en">English</a></p>
        <p><a href="https://openaitx.github.io/view.html?user=tau-yihouxiang&project=EX-4D&lang=zh-CN">简体中文</a></p>
        <p><a href="https://openaitx.github.io/view.html?user=tau-yihouxiang&project=EX-4D&lang=zh-TW">繁體中文</a></p>
        <p><a href="https://openaitx.github.io/view.html?user=tau-yihouxiang&project=EX-4D&lang=ja">日本語</a></p>
        <p><a href="https://openaitx.github.io/view.html?user=tau-yihouxiang&project=EX-4D&lang=ko">한국어</a></p>
        <p><a href="https://openaitx.github.io/view.html?user=tau-yihouxiang&project=EX-4D&lang=hi">हिन्दी</a></p>
        <p><a href="https://openaitx.github.io/view.html?user=tau-yihouxiang&project=EX-4D&lang=th">ไทย</a></p>
        <p><a href="https://openaitx.github.io/view.html?user=tau-yihouxiang&project=EX-4D&lang=fr">Français</a></p>
        <p><a href="https://openaitx.github.io/view.html?user=tau-yihouxiang&project=EX-4D&lang=de">Deutsch</a></p>
        <p><a href="https://openaitx.github.io/view.html?user=tau-yihouxiang&project=EX-4D&lang=es">Español</a></p>
        <p><a href="https://openaitx.github.io/view.html?user=tau-yihouxiang&project=EX-4D&lang=it">Itapano</a></p>
        <p><a href="https://openaitx.github.io/view.html?user=tau-yihouxiang&project=EX-4D&lang=ru">Русский</a></p>
        <p><a href="https://openaitx.github.io/view.html?user=tau-yihouxiang&project=EX-4D&lang=pt">Português</a></p>
        <p><a href="https://openaitx.github.io/view.html?user=tau-yihouxiang&project=EX-4D&lang=nl">Nederlands</a></p>
        <p><a href="https://openaitx.github.io/view.html?user=tau-yihouxiang&project=EX-4D&lang=pl">Polski</a></p>
        <p><a href="https://openaitx.github.io/view.html?user=tau-yihouxiang&project=EX-4D&lang=ar">العربية</a></p>
        <p><a href="https://openaitx.github.io/view.html?user=tau-yihouxiang&project=EX-4D&lang=fa">فارسی</a></p>
        <p><a href="https://openaitx.github.io/view.html?user=tau-yihouxiang&project=EX-4D&lang=tr">Türkçe</a></p>
        <p><a href="https://openaitx.github.io/view.html?user=tau-yihouxiang&project=EX-4D&lang=vi">Tiếng Việt</a></p>
        <p><a href="https://openaitx.github.io/view.html?user=tau-yihouxiang&project=EX-4D&lang=id">Bahasa Indonesia</a></p>
      </div>
    </div>
  </details>
</div>

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
git clone https://github.com/tau-yihouxiang/EX-4D.git
cd EX-4D

# Create conda environment
conda create -n ex4d python=3.10
conda activate ex4d
# Install PyTorch (2.x recommended)
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
# Install Nvdiffrast
pip install git+https://github.com/NVlabs/nvdiffrast.git
# Install dependencies and diffsynth
pip install -e .
# Install depthcrafter for depth estimation. (Follow DepthCrafter's installing instruction for checkpoints preparation.)
git clone https://github.com/Tencent/DepthCrafter.git
```

### Download Pretrained Model
```bash
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./models/Wan-AI
huggingface-cli download yihouxiang/EX-4D --local-dir ./models/EX-4D
```

### Example Usage
#### 1. DW-Mesh Reconstruction
```bash
# --cam 180 (30 / 60 / 90 / zoom_in / zoom_out )
python recon.py --input_video examples/flower/input.mp4 --cam 180 --output_dir outputs/flower --save_mesh
```
#### 2. EX-4D Generation (48GB VRAM required)
```bash
python generate.py --color_video outputs/flower/color_180.mp4 --mask_video outputs/flower/mask_180.mp4 --output_video outputs/flower/output.mp4
```

<table>
<tr>
<td width="45%" align="center">
<img src="examples/flower/input.gif" width="100%">
<br><b>Input Video</b>
</td>
<td align="center">
<div style="font-size: 2em; color: #4A90E2; padding: 0 0px;">
  ➜
</div>
</td>
<td width="45%" align="center">
<img src="examples/flower/output.gif" width="100%">
<br><b>Output Video</b>
</td>
</tr> 
</table>

<!-- ## 📊 Performance

### Quantitative Results

| Method | FID (Extreme) ↓ | FVD (Extreme) ↓ | VBench Score ↑ |
|--------|-----------------|-----------------|----------------|
| ReCamMaster | 64.68 | 943.45 | 0.434 |
| TrajectoryCrafter | 65.33 | 893.80 | 0.447 |
| TrajectoryAttention | 62.49 | 912.14 | 0.389 |
| **EX-4D (Ours)** | **55.42** | **823.61** | **0.450** | -->

### User Study Results

- **70.7%** of participants preferred EX-4D over baseline methods
- Superior performance in physical consistency and extreme viewpoint quality
- Significant improvement as camera angles become more extreme


## 🎯 Applications

- **🎮 Gaming**: Create immersive 3D game cinematics from 2D footage
- **🎬 Film Production**: Generate novel camera angles for post-production
- **🥽 VR/AR**: Create free-viewpoint video experiences
- **📱 Social Media**: Generate dynamic camera movements for content creation
- **🏢 Architecture**: Visualize spaces from multiple viewpoints

<!-- ## 📈 Benchmarks -->

<!-- ### Viewpoint Range Evaluation

| Range | Small (0°→30°) | Large (0°→60°) | Extreme (0°→90°) | Full (-90°→90°) |
|-------|----------------|----------------|------------------|-----------------|
| FID Score | 44.19 | 50.30 | 55.42 | - |
| Performance Gap | +9.1% better | +8.9% better | +11.3% better | +15.5% better | -->

<!-- *Performance gap compared to the second-best method in each category.* -->

## ⚠️ Limitations

- **Depth Dependency**: Performance relies on monocular depth estimation quality
- **Computational Cost**: Requires significant computation for high-resolution videos
- **Reflective Surfaces**: Challenges with reflective or transparent materials

## 🔮 Future Work
- [ ] Real-time inference optimization (3DGS / 4DGS)
- [ ] Support for higher resolutions (1K, 2K)
- [ ] Neural mesh refinement techniques

## 🙏 Acknowledgments

We would like to thank the [DiffSynth-Studio v1.1.1](https://github.com/modelscope/DiffSynth-Studio/tree/v1.1.1) project for providing the foundational diffusion framework.

## 📚 Citation

If you find our work useful, please consider citing:

```bibtex
@misc{hu2025ex4dextremeviewpoint4d,
      title={EX-4D: EXtreme Viewpoint 4D Video Synthesis via Depth Watertight Mesh}, 
      author={Hu, Tao and Peng, Haoyang and Liu, Xiao and Ma, Yuewen},
      year={2025},
      eprint={2506.05554},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.05554}
}
```
