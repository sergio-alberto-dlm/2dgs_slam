import torch 
import numpy as np 
import cv2 
from utils.logging_utils import Log
from submodules.unimatch.unimatch.unimatch import UniMatch

def load_model(model_path:str, DEVICE="cuda:0"):
    """load the pre-trained model""" 
    model = UniMatch(
        feature_channels=128,    
        num_scales=1, 
        upsample_factor=8,
        num_head=1,
        ffn_dim_expansion=4,
        num_transformer_layers=6,
        reg_refine=False,
        task='flow'
    ).to(DEVICE)
    ckpt = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(ckpt['model'], strict=True)
    model.eval()
    Log("Model loaded")
    return model 

def get_cam_coords(intrins_inv, depth):
    """camera coordinates from intrins_inv and depth

    NOTE: intrins_inv should be a torch tensor of shape (B, 3, 3)
    NOTE: depth should be a torch tensor of shape (B, 1, H, W)
    NOTE: top-left is (0,0)
    """
    assert torch.is_tensor(intrins_inv) and intrins_inv.ndim == 3
    assert torch.is_tensor(depth) and depth.ndim == 4
    assert intrins_inv.dtype == depth.dtype
    assert intrins_inv.device == depth.device
    B, _, H, W = depth.size()

    u_range = (
        torch.arange(W, dtype=depth.dtype, device=depth.device).view(1, W).expand(H, W)
    )  # (H, W)
    v_range = (
        torch.arange(H, dtype=depth.dtype, device=depth.device).view(H, 1).expand(H, W)
    )  # (H, W)
    ones = torch.ones(H, W, dtype=depth.dtype, device=depth.device)
    pixel_coords = (
        torch.stack((u_range, v_range, ones), dim=0).unsqueeze(0).repeat(B, 1, 1, 1)
    )  # (B, 3, H, W)
    pixel_coords = pixel_coords.view(B, 3, H * W)  # (B, 3, H*W)

    cam_coords = intrins_inv.bmm(pixel_coords).view(B, 3, H, W)
    cam_coords = cam_coords * depth
    return cam_coords

def project_and_loss(pts3d_curr, dst_pixels, K):
    """
    pts3d_curr: (N,3) in *current* camera frame
    dst_pixels: (N,2) observed matches from flow
    K: (3,3) intrinsics

    Returns scalar reprojection loss.
    """
    # project: x = K [X/Z, Y/Z, 1]
    X, Y, Z = pts3d_curr[:,0], pts3d_curr[:,1], pts3d_curr[:,2].clamp(min=1e-3)
    uv_proj = torch.stack([X/Z, Y/Z, torch.ones_like(Z)], dim=1)  # (N,3)
    uv = (K @ uv_proj.t()).t()[:, :2]                          # (N,2)
    # L2
    loss = (uv - dst_pixels).pow(2).sum(dim=1).mean()
    return loss

def compute_flow(model, view_left, view_right):
    """
        Computes the flow estimated with the pre-trained model 
    """
    img1 = view_left.original_image.unsqueeze(0)
    img2 = view_right.original_image.unsqueeze(0)
    with torch.no_grad():
        out = model(img1, img2,
                    attn_type='swin',
                    attn_splits_list=[2],
                    corr_radius_list=[-1],
                    prop_radius_list=[-1],
                    num_reg_refine=1,
                    task='flow')
    flow = out['flow_preds'][-1]  # (1,2,H,W)
    B, C, H, W = flow.shape
    # convert to pixel coords
    flow = flow[0].permute(1,2,0)  # (H,W,2)
    return flow 

def compute_induce_flow(viewpoint):
    pass

def compute_flow_matches(model, view_left, view_right, DEVICE):
    """
        Runs your UniMatch model and returns two FloatTensors of shape (N,2):
        src_pixels: pixel coords in frame1 (x,y)
        dst_pixels: corresponding pixel coords in frame2 (x,y)
    """
    img1 = view_left.original_image.unsqueeze(0)
    img2 = view_right.original_image.unsqueeze(0)
    with torch.no_grad():
        out = model(img1, img2,
                    attn_type='swin',
                    attn_splits_list=[2],
                    corr_radius_list=[-1],
                    prop_radius_list=[-1],
                    num_reg_refine=1,
                    task='flow')
    flow = out['flow_preds'][-1]  # (1,2,H,W)
    B, C, H, W = flow.shape
    # convert to pixel coords
    flow = flow[0].permute(1,2,0)             # (H,W,2)
    xs = torch.arange(W, device=flow.device)
    ys = torch.arange(H, device=flow.device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    pts1 = torch.stack([grid_x, grid_y], dim=-1).to(DEVICE).float()  # (H,W,2)
    pts2 = pts1 + flow
    # round & mask
    pts2_ri = pts2.round()
    valid = (
        (pts2_ri[...,0] >= 0) & (pts2_ri[...,0] < W) &
        (pts2_ri[...,1] >= 0) & (pts2_ri[...,1] < H)
    )
    src = pts1[valid]     # (N,2)
    dst = pts2_ri[valid]
    return src, dst, flow, valid 

def draw_flow_matches(view_left, view_right, src_pts, dst_pts, max_matches=1000, 
                      point_color=(0,255,0), line_color=(0,255,0), thickness=1):
    """
    img1, img2: H×W×3 BGR images (uint8)
    src_pts: (N,2) array of (x,y) in img1
    dst_pts: (N,2) array of (x,y) in img2
    max_matches: randomly sample up to this many lines (for speed/clarity)
    """
    img1 = (view_left.original_image.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint16)
    img2 = (view_right.original_image.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint16)
    h, w = img1.shape[:2]
    # stack images side by side
    vis = np.concatenate([img1, img2], axis=1).copy()
    
    N = src_pts.shape[0]
    if N > max_matches:
        idxs = np.random.choice(N, max_matches, replace=False)
        src_pts = src_pts[idxs]
        dst_pts = dst_pts[idxs]
    
    for (x1, y1), (x2, y2) in zip(src_pts, dst_pts):
        # draw circle at source
        cv2.circle(vis, (int(x1), int(y1)), 2, point_color, -1)
        # draw circle at dest (offset x by w)
        cv2.circle(vis, (int(x2) + w, int(y2)), 2, point_color, -1)
        # draw line
        cv2.line(vis, (int(x1), int(y1)), (int(x2) + w, int(y2)), line_color, thickness)
    
    return vis
 