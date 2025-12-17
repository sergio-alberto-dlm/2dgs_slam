import torch
from torch import nn
import sys
import os

from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2
from utils.slam_utils import image_gradient, image_gradient_mask
import torch.nn.functional as F
from utils.normal_utils import intrins_to_intrins_inv, get_cam_coords, d2n_tblr
import numpy as np


# Global model cache for depth estimation
_DEPTH_MODEL_CACHE = None


def get_depth_model(device="cuda:0"):
    """
    Load and cache the DepthAnything3 model.
    Returns a singleton instance to avoid reloading the model multiple times.
    """
    global _DEPTH_MODEL_CACHE
    
    if _DEPTH_MODEL_CACHE is None:
        try:
            # Add depth_anything_v3 src to path
            depth_anything_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'submodules', 'depth_anything_v3', 'src'
            )
            if depth_anything_path not in sys.path:
                sys.path.insert(0, depth_anything_path)
            
            from depth_anything_3.api import DepthAnything3
            
            print("Loading DepthAnything3 model...")
            model = DepthAnything3.from_pretrained("depth-anything/da3-base")
            model = model.to(device)
            model.eval()
            _DEPTH_MODEL_CACHE = model
            print(f"DepthAnything3 model loaded on {device}")
        except Exception as e:
            print(f"Warning: Could not load DepthAnything3 model: {e}")
            _DEPTH_MODEL_CACHE = None
    
    return _DEPTH_MODEL_CACHE

class Camera(nn.Module):
    def __init__(
        self,
        uid,
        color,
        depth,
        gt_T,
        projection_matrix,
        fx,
        fy,
        cx,
        cy,
        fovx,
        fovy,
        image_height,
        image_width,
        device="cuda:0",
        fid=None,
    ):
        super(Camera, self).__init__()
        self.uid = uid
        self.fid = fid
        self.device = device

        self.T = torch.eye(4, device=device).to(torch.float32)
        self.T_gt = gt_T.to(device=device).to(torch.float32).clone()

        self.original_image = color
        self.depth = depth
        self.grad_mask = None

        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.FoVx = fovx
        self.FoVy = fovy
        self.image_height = image_height
        self.image_width = image_width

        self.gp_mean = None
        self.gp_var = None 
        self.residual_thresh = None 

        # lie algebra minimal parametrization for camera parameters 
        self.cam_rot_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )
        self.cam_trans_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )

        self.exposure_a = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )
        self.exposure_b = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )

        self.projection_matrix = projection_matrix.to(device=device)

        self.intrins = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
        self.intrins_inv = intrins_to_intrins_inv(self.intrins).float().unsqueeze(0).to(0)

    @staticmethod
    def init_from_dataset(dataset, idx, projection_matrix):
        gt_color, gt_depth, gt_pose = dataset[idx]

        return Camera(
            idx,
            gt_color,
            gt_depth,
            gt_pose,
            projection_matrix,
            dataset.fx,
            dataset.fy,
            dataset.cx,
            dataset.cy,
            dataset.fovx,
            dataset.fovy,
            dataset.height,
            dataset.width,
            device=dataset.device,
            fid=float(idx) / len(dataset),
        )

    @staticmethod
    def init_from_gui(uid, T, FoVx, FoVy, fx, fy, cx, cy, H, W):
        projection_matrix = getProjectionMatrix2(
            znear=0.01, zfar=100.0, fx=fx, fy=fy, cx=cx, cy=cy, W=W, H=H
        ).transpose(0, 1)
        return Camera(
            uid, None, None, T, projection_matrix, fx, fy, cx, cy, FoVx, FoVy, H, W
        )

    @property
    def world_view_transform(self):
        return self.T.transpose(0, 1).to(device=self.device)

    @property
    def full_proj_transform(self):
        return (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)

    @property
    def camera_center(self):
        return self.world_view_transform  # TODO: Need to invert for high order SHs by inverse_t(self.world_view_transform).

    def compute_monocular_depth(self, process_res=504, scale_adjustment=1.0, init=False):
        """
        Compute monocular depth estimation using DepthAnything3 if gt_depth is not available.
        
        Args:
            process_res: Resolution for depth model processing (default: 504)
            scale_adjustment: Scale factor to adjust depth values (default: 1.0)
            init: Whether this is the initialization frame (default: False)
        
        Returns:
            If init=True: (depth, extrinsics, intrinsics) tuple
            If init=False: depth only
        """
        if self.depth is not None:
            # Ground truth depth already available, no need to estimate
            if init:
                return self.depth, None, None
            return self.depth
        
        # Get the depth model
        model = get_depth_model(device=self.device)
        if model is None:
            print("Warning: Depth model not available. Cannot compute monocular depth.")
            if init:
                return None, None, None
            return None
        
        try:
            # Convert image from torch tensor (C, H, W) [0,1] to numpy (H, W, C) [0,255]
            image_np = (self.original_image.permute(1, 2, 0).cpu().numpy() * 255.0).astype('uint8')
            
            # Preprocess images
            if init:
                extrinsics = None
                intrinsics = None
            else:
                extrinsics = self.T.clone().unsqueeze(0).cpu().numpy()
                intrinsics = self.intrins.clone().unsqueeze(0).cpu().numpy()
            imgs_cpu, extrinsics, intrinsics = model._preprocess_inputs(
                image=[image_np],
                extrinsics=extrinsics,
                intrinsics=intrinsics,
                process_res=process_res,
                process_res_method="upper_bound_resize"
            )
            
            # Prepare tensors for model
            imgs, ex_t, in_t = model._prepare_model_inputs(imgs_cpu, extrinsics, intrinsics)
            
            # Normalize extrinsics
            ex_t_norm = model._normalize_extrinsics(ex_t.clone() if ex_t is not None else None)
            
            # Run model forward pass
            export_feat_layers = []
            with torch.no_grad():
                raw_output = model._run_model_forward(
                    imgs, ex_t_norm, in_t, export_feat_layers,
                )
            
            # Convert raw output to prediction
            prediction = model._convert_to_prediction(raw_output)
            
            # Align prediction to extrinsics/intrinsics
            prediction = model._align_to_input_extrinsics_intrinsics(
                prediction=prediction, extrinsics=None, intrinsics=None
            )
            
            # Extract depth map from prediction
            # The prediction typically contains the depth in the first element
            depth_map = prediction.depth[0]
            
            # Move depth to CPU immediately to free GPU memory
            if isinstance(depth_map, torch.Tensor):
                depth_map = depth_map.cpu()
            
            # Clean up intermediate tensors
            del raw_output, prediction, imgs, ex_t, in_t, ex_t_norm, imgs_cpu
            torch.cuda.empty_cache()
            
            # Resize depth map to original image resolution if needed
            if isinstance(depth_map, torch.Tensor):
                depth_h, depth_w = depth_map.shape[-2:]
            else:
                depth_h, depth_w = depth_map.shape[:2]
            
            if depth_h != self.image_height or depth_w != self.image_width:
                # Convert to tensor if numpy
                if isinstance(depth_map, np.ndarray):
                    depth_map = torch.from_numpy(depth_map)
                
                # Ensure depth is on CPU for resizing to save GPU memory
                depth_map = depth_map.cpu()
                
                # Ensure proper shape for interpolation [1, 1, H, W]
                if depth_map.dim() == 2:
                    depth_map = depth_map.unsqueeze(0).unsqueeze(0)
                elif depth_map.dim() == 3:
                    depth_map = depth_map.unsqueeze(0)
                
                # Resize using bilinear interpolation
                depth_map = F.interpolate(
                    depth_map,
                    size=(self.image_height, self.image_width),
                    mode='bilinear',
                    align_corners=False
                )
                
                # Remove batch and channel dimensions
                depth_map = depth_map.squeeze(0).squeeze(0)
                
                # Convert to numpy to keep on CPU
                depth_map = depth_map.numpy()
            
            # Apply scale adjustment if needed
            if scale_adjustment != 1.0:
                if isinstance(depth_map, torch.Tensor):
                    depth_map = depth_map.cpu().numpy() * scale_adjustment
                else:
                    depth_map = depth_map * scale_adjustment
            else:
                # Ensure depth is numpy array on CPU
                if isinstance(depth_map, torch.Tensor):
                    depth_map = depth_map.cpu().numpy()
            
            # Return based on init flag (depth is now always numpy on CPU)
            if init:
                # Note: extrinsics and intrinsics from prediction are already on CPU
                return depth_map, None if not hasattr(prediction, 'extrinsics') else prediction.extrinsics, \
                       None if not hasattr(prediction, 'intrinsics') else prediction.intrinsics
            return depth_map
            
        except Exception as e:
            print(f"Error computing monocular depth: {e}")
            import traceback
            traceback.print_exc()
            if init:
                return None, None, None
            return None

    def update_intrinsics(self, intrinsics_matrix):
        """
        Update camera intrinsic parameters from a 3x3 intrinsics matrix.
        
        Args:
            intrinsics_matrix: 3x3 numpy array or torch tensor with camera intrinsics
        """
        if intrinsics_matrix is None:
            return
        
        # Convert to numpy if torch tensor
        if isinstance(intrinsics_matrix, torch.Tensor):
            intrinsics_matrix = intrinsics_matrix.cpu().numpy()
        
        # Extract intrinsic parameters
        self.fx = float(intrinsics_matrix[0, 0])
        self.fy = float(intrinsics_matrix[1, 1])
        self.cx = float(intrinsics_matrix[0, 2])
        self.cy = float(intrinsics_matrix[1, 2])
        
        # Update FoV based on new focal lengths
        from gaussian_splatting.utils.graphics_utils import focal2fov
        self.FoVx = focal2fov(self.fx, self.image_width)
        self.FoVy = focal2fov(self.fy, self.image_height)
        
        # Update intrinsics matrices
        self.intrins = torch.tensor([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])
        self.intrins_inv = intrins_to_intrins_inv(self.intrins).float().unsqueeze(0).to(self.device)
        
        # Update projection matrix
        from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2
        self.projection_matrix = getProjectionMatrix2(
            znear=0.01, zfar=100.0, 
            fx=self.fx, fy=self.fy, 
            cx=self.cx, cy=self.cy, 
            W=self.image_width, H=self.image_height
        ).transpose(0, 1).to(device=self.device)
        
        print(f"Updated camera intrinsics: fx={self.fx:.2f}, fy={self.fy:.2f}, cx={self.cx:.2f}, cy={self.cy:.2f}")

    def compute_grad_mask(self, config):
        """
            Function to compute the grad mask and the normals initialization
        """
        edge_threshold = config["Training"]["edge_threshold"]

        gray_img = self.original_image.mean(dim=0, keepdim=True)
        gray_grad_v, gray_grad_h = image_gradient(gray_img)
        mask_v, mask_h = image_gradient_mask(gray_img)
        gray_grad_v = gray_grad_v * mask_v
        gray_grad_h = gray_grad_h * mask_h
        img_grad_intensity = torch.sqrt(gray_grad_v**2 + gray_grad_h**2)

        if config["Dataset"]["type"] == "replica":
            size = 32
            multiplier = edge_threshold
            _, h, w = self.original_image.shape
            I = img_grad_intensity.unsqueeze(0)
            I_unf = F.unfold(I, size, stride=size)
            median_patch, _ = torch.median(I_unf, dim=1, keepdim=True)
            mask = (I_unf > (median_patch * multiplier)).float()
            I_f = F.fold(mask, I.shape[-2:], size, stride=size).squeeze(0)
            self.grad_mask = I_f
        else:
            median_img_grad_intensity = img_grad_intensity.median()
            self.grad_mask = (
                img_grad_intensity > median_img_grad_intensity * edge_threshold
            )

        gt_image = self.original_image.cuda()
        _, h, w = self.original_image.cuda().shape
        mask_shape = (1, h, w)
        rgb_boundary_threshold = 0.05
        rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(
            *mask_shape
        ) 
        self.rgb_pixel_mask = rgb_pixel_mask * self.grad_mask

        self.rgb_pixel_mask = rgb_pixel_mask
        self.rgb_pixel_mask_mapping = rgb_pixel_mask

        if self.depth is not None:
            self.gt_depth = torch.from_numpy(self.depth).to(
                dtype=torch.float32, device=self.device
            )[None]

            depth = self.gt_depth.unsqueeze(0)
            points = get_cam_coords(self.intrins_inv, depth)
            normal, valid_mask = d2n_tblr(points, d_min=1e-3, d_max=1000.0)
            normal = normal * valid_mask
            self.normal = normal
            self.normal_raw = self.normal.squeeze(0).permute(1, 2, 0).cpu().numpy()
            self.mask = valid_mask 

            if self.mask is not None:
                self.rgb_pixel_mask = self.rgb_pixel_mask * self.mask
                self.rgb_pixel_mask_mapping = self.rgb_pixel_mask_mapping

    def clean(self):
        self.original_image = None
        self.depth = None
        self.grad_mask = None

        self.cam_rot_delta = None
        self.cam_trans_delta = None

        self.exposure_a = None
        self.exposure_b = None

        self.rgb_pixel_mask = None
        self.rgb_pixel_mask_mapping = None
        self.gt_depth = None
        self.normal = None
        self.normal_raw = None

class CameraMsg:
    def __init__(self, Camera=None, uid=None, T=None, T_gt=None, fid=None):
        if Camera is not None:
            self.uid = Camera.uid
            self.T = Camera.T
            self.T_gt = Camera.T_gt 
            self.fid = Camera.fid
        else:
            self.uid = uid 
            self.T = T
            self.T_gt = T_gt 
            self.fid = fid