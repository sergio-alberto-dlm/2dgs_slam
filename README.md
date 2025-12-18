# 2D Gaussian Splatting SLAM System 🚀

This repository implements a **2D Gaussian Splatting SLAM system**. It utilizes 2D Gaussian Splatting as the core map representation to achieve high-fidelity reconstruction and tracking. 

### Key Features
* **2D Gaussian Map Representation:** Efficient and high-quality geometry representation.
* **Interactive Visualization:** Implements a light [Rerun](https://rerun.io/) demo for real-time visualization.
* **Monocular Depth Estimation (Experimental):** Includes a feature to compute monocular depth maps conditioned on camera poses estimated by the tracking module, utilizing **Depth Anything V3**.

> **⚠️ Note:** The Depth Anything V3 integration is an **experimental feature**. Results may vary, and we encourage users to open an issue to report any potential bugs or unexpected behavior.

### Theoretical Resources
For those interested in the underlying mechanics of this project:
* **Technical report:** Please check the file [`2d_gaussian_splatting_based_camera_localization.pdf`](./2d_gaussian_splatting_based_camera_localization.pdf) located in this repository.
* **Blog Post:** For a friendly introduction to the concepts, check out: [2D Gaussian Splatting: from pixels to geometry, part 1](https://medium.com/@sergio.deleon_41219/2d-gaussian-splatting-from-pixels-to-geometry-part-1-b08763fbfefe).

---

## 🛠️ Installation

### 1. Clone the Repository
Since this project relies on external submodules, please clone recursively:

```bash
git clone --recursive https://github.com/sergio-alberto-dlm/2dgs_slam
cd 2dgs_slam
```

### 2. Create Environment & Install Core Dependencies
First, set up the Conda environment and install the core PyTorch and CUDA dependencies.

```bash
conda create -n 2dgs_slam python=3.11
conda activate 2dgs_slam

# Install PyTorch with CUDA support according to your system 
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=11.8 -c pytorch -c nvidia

# We need to install specific versions of PyTorch3D and LieTorch without build isolation to ensure compatibility with the system environment.

# Install PyTorch3D
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git"

# Install LieTorch 
pip install --no-build-isolation "git+https://github.com/princeton-vl/lietorch.git@0fa9ce8ffca86d985eca9e189a99690d6f3d4df6"
```

### 3. Install Remaining Dependencies
Finally, install the rest of the required packages using the provided YAML file:

```bash 
conda env update --file environment.yaml
```

### 📂 Datasets
To test the system, please download the following datasets:

**TUM-RGBD Dataset**:

* Download from: [TUM Computer Vision Group](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download)

**Replica Dataset**:

* Download from: [Replica Dataset GitHub](https://github.com/facebookresearch/Replica-Dataset)

### 🚀 Usage

**Configuration**

Before running, check the base_config.yaml file. You can modify parameters here to switch between evaluating trajectory vs. appearance, or to enable the Rerun light demo.

**Running on Standard Datasets**

To run the main SLAM script on a TUM dataset sequence (e.g., fr1_desk):

```bash 
python surfelSLAM.py --config configs/rgbd/tum/fr1_desk.yaml
```

### Running on "In-the-Wild" Videos
If you want to play with your own data:

1. Place your video file in the demo_videos folder.

2. Run the video configuration (depth maps will be estimated automatically using Depth Anything V3):

```bash 
python surfelSLAM.py --config configs/rgb/video/iphone_flowers.yaml
```

### 🙏 Acknowledgments
This project is built upon excellent open-source work. We would like to thank the authors of the following projects:

* [Differential Surfel Rasterization with Pose](https://github.com/muskie82/diff-surfel-rasterization-with-pose/tree/13f0e3c5a31ed004b1eba907a5cc1922052de553)

* [2D Gaussian Splatting](https://github.com/hbb1/2d-gaussian-splatting)

* [Rerun.io](https://rerun.io/)

* [Depth Anything V3](https://github.com/ByteDance-Seed/Depth-Anything-3)