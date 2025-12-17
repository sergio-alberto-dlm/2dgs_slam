# RGB-D SLAM with 2D Gaussian Splatting and Gaussian Processes 🚀

Welcome to our RGB-D SLAM project! This repository implements a full pipeline for real-time camera tracking and dense mapping using **2D Gaussian Splatting (2DGS)** as the map representation and a **Gaussian Process (GP)** model to handle dynamic objects and distractors.  

The system is built in Python using **PyTorch** and **CUDA**, and it demonstrates how to fuse rendering-based SLAM with probabilistic modelling for robust performance in dynamic scenes.

---
## Demo 

balloon scene of RGBD-Bonn dataset

![Demo GIF](assets/demo.gif)

---

## 🛠️ Installation

We recommend using **conda** to manage dependencies.

```bash
# Clone recursively to include all submodules
git clone --recursive https://github.com/sergio-alberto-dlm/DynamicGS-SLAM.git
cd DynamicGS-SLAM

# Create environment (Python 3.9)
conda create -n slam_env python=3.9
conda activate slam_env

# Install PyTorch with CUDA support
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia

# Install CUDA-dependent libraries
conda install pytorch3d -c pytorch3d
pip install --no-build-isolation git+https://github.com/princeton-vl/lietorch.git

# Install the remaining requirements
pip install -r requirements.txt
```

That’s it!  You should now be ready to run the demo. If you experience issues with CUDA or dependencies, please submit an issue. 

## 🧭 Overview of the SLAM Pipeline

Our pipeline implements a differentiable SLAM system that uses 2D Gaussian Splatting to represent the map. 2DGS collapses a 3D volume into a set of oriented planar Gaussian disks, providing view‑consistent geometry and enabling real‑time rendering. Compared to standard 3D Gaussian splatting, 2DGS includes depth distortion and normal consistency terms and is optimized using custom CUDA kernels for fast and noise‑free rendering.

The pipeline is split into two main modules that run in parallel using PyTorch’s multiprocessing utilities:

### 🔄 Front‑End (Tracking)

* Processes incoming frames sequentially to estimate the camera pose.

* Uses the current Gaussian map directly for pose optimization via analytic gradients.

* Maintains a local active map for efficient tracking and loop closure.

### 🗺️ Back‑End (Mapping)

Performs training, densification and refinement of the map.

Updates the set of Gaussian parameters using the differentiable 2DGS renderer and backpropagation.

Runs concurrently with the front‑end in a separate process, sharing data through shared memory for efficiency.

The result is a high‑quality reconstruction with accurate camera trajectories. Similar to 2DGS‑SLAM, our method leverages the depth‑consistent rendering of 2DGS to achieve superior tracking accuracy and consistent global maps compared to alternative rendering‑based SLAM systems.

## 🌪 Handling Dynamics and Distractors

Real‑world scenes are often dynamic; moving objects or distractors can corrupt the camera pose estimation and map optimization. Previous work such as WildGS‑SLAM uses a shallow multi‑layer perceptron (MLP) to predict a per‑pixel uncertainty mask from pre‑trained vision features (e.g., DINOv2), which guides dynamic object removal during tracking and mapping
openaccess.thecvf.com
. While effective, MLPs require careful training and might not capture complex uncertainty patterns.

🌾 Our Probabilistic Approach with Gaussian Processes

We model the residual between observed colors and the rendered 2DGS map as a Gaussian Process (GP) conditioned on semantic features extracted by FiT3D. A GP defines a distribution over functions: any finite collection of function values has a multivariate normal distribution. Using a GP allows us to represent uncertainty in a principled way and to naturally capture correlations between observations.

Variational inference: We employ a sparse GP with inducing points and optimize the variational evidence lower bound (ELBO). Variational inference turns Bayesian inference into an optimization problem by maximizing a lower bound on the log marginal likelihood, allowing us to learn both the inducing points and hyperparameters jointly.

Feature extraction with FiT3D: FiT3D fine‑tunes DINOv2 features by lifting them into a 3D Gaussian representation, enriching the features with geometry‑aware information and improving semantic segmentation and depth estimation performance. These semantically rich features are concatenated with the photometric residual to form the training dataset for the GP.

**Greedy max‑variance point selection**: At each optimization step, we select points with the highest predictive variance from the GP to annotate or refine. This greedy strategy focuses computational resources on the most uncertain regions, improving robustness to distractors.

Probabilistic mask: The predictive mean and variance from the GP produce a distractor mask that weights residuals during pose optimization and updates the mapping loss. Regions with high variance (likely dynamics) contribute less to the gradient, improving stability and reconstruction quality.

This approach provides a theoretically grounded alternative to MLP‑based uncertainty estimation and can adapt to complex dynamics without manual tuning.

## ⚙️ Technical Highlights

Multiprocessing with PyTorch: Both the front‑end and back‑end modules run in parallel processes using torch.multiprocessing, which seamlessly shares tensors between processes with minimal overhead.

CUDA‑accelerated rendering: Our 2DGS renderer builds upon the official 3DGS codebase and uses custom CUDA kernels to render oriented Gaussian disks, compute depth distortion and normal maps, and support real‑time gradients.

Full PyTorch implementation: The entire SLAM system, including training, tracking, and GP inference, is implemented in PyTorch for flexibility and GPU acceleration.

Gaussian Process with GPyTorch: We leverage the GPyTorch library, which provides efficient and modular GP implementations with GPU support. The library integrates seamlessly with PyTorch and enables scalable variational GPs on large datasets.

Visualization with Rerun: For quick debugging and demonstrations, we integrate the Rerun SDK, an open‑source log handling and visualization tool for spatial and embodied AI. Rerun allows us to stream, store and visualize camera trajectories, Gaussian maps and uncertainty masks with minimal code
rerun.io.

## 🙏 Acknowledgements

We are grateful to the creators of the following projects, which have significantly influenced our work:

* **4DTAM: Non‑Rigid Tracking and Mapping via Dynamic Surface Gaussians** – Introduced a pioneering SLAM technique using Gaussian surface primitives and differentiable rendering to handle non‑rigid deformations. Their insights into dynamic surface reconstruction inspired our use of Gaussians and deepened our understanding of the depth signals.

* **2D Gaussian Splatting (2DGS)** – Provided the core map representation used in this project; their perspective‑accurate splatting and fast CUDA implementation enabled real‑time rendering and optimization.

* **2DGS‑SLAM** – Demonstrated the effectiveness of 2DGS for globally consistent RGB‑D SLAM and motivated our choice of representation and loop closure strategies.

* **WildGS‑SLAM** – Proposed an MLP‑based uncertainty mask for dynamic environments; our Gaussian Process approach offers an alternative probabilistic method inspired by their work.

* **FiT3D** – Provided geometry‑aware feature extraction that we leverage to supply rich semantics to the GP.

* **GPyTorch** – An excellent open‑source library for scalable Gaussian Processes on GPUs.

* **Rerun** – For the convenient visualization and logging tools that help us debug and demonstrate the pipeline.

We thank all these projects and their authors for open‑sourcing their work!