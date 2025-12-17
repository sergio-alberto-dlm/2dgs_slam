import numpy as np
import torch
import lietorch


def rt2mat(R, T):
    mat = np.eye(4)
    mat[0:3, 0:3] = R
    mat[0:3, 3] = T
    return mat


def skew_sym_mat(x):
    device = x.device
    dtype = x.dtype
    ssm = torch.zeros(3, 3, device=device, dtype=dtype)
    ssm[0, 1] = -x[2]
    ssm[0, 2] = x[1]
    ssm[1, 0] = x[2]
    ssm[1, 2] = -x[0]
    ssm[2, 0] = -x[1]
    ssm[2, 1] = x[0]
    return ssm


def SO3_exp(theta):
    device = theta.device
    dtype = theta.dtype

    W = skew_sym_mat(theta)
    W2 = W @ W
    angle = torch.norm(theta)
    I = torch.eye(3, device=device, dtype=dtype)
    if angle < 1e-5:
        return I + W + 0.5 * W2
    else:
        return (
            I
            + (torch.sin(angle) / angle) * W
            + ((1 - torch.cos(angle)) / (angle**2)) * W2
        )


def V(theta):
    dtype = theta.dtype
    device = theta.device
    I = torch.eye(3, device=device, dtype=dtype)
    W = skew_sym_mat(theta)
    W2 = W @ W
    angle = torch.norm(theta)
    if angle < 1e-5:
        V = I + 0.5 * W + (1.0 / 6.0) * W2
    else:
        V = (
            I
            + W * ((1.0 - torch.cos(angle)) / (angle**2))
            + W2 * ((angle - torch.sin(angle)) / (angle**3))
        )
    return V


def SE3_exp(tau):
    dtype = tau.dtype
    device = tau.device

    rho = tau[:3]
    theta = tau[3:]
    R = SO3_exp(theta)
    t = V(theta) @ rho

    T = torch.eye(4, device=device, dtype=dtype)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def inverse(T):
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = torch.eye(4, device=T.device, dtype=T.dtype)
    T_inv[:3, :3] = R.t()
    T_inv[:3, 3] = -R.t() @ t
    return T_inv


def inverse_t(T):
    return -T[:3, :3].t() @ T[:3, 3]


def update_pose(camera, converged_threshold=1e-4):
    tau = torch.cat([camera.cam_trans_delta, camera.cam_rot_delta], axis=0)
    T_w2c = camera.T
    new_w2c = lietorch.SE3.exp(tau).matrix() @ T_w2c
    converged = (tau**2).sum() < (converged_threshold**2)
    camera.T = new_w2c
    camera.cam_rot_delta.data.fill_(0)
    camera.cam_trans_delta.data.fill_(0)

    return converged


def quat2rotmat_batch(quaternions):
    """
    Convert a batch of quaternions to a batch of 3x3 rotation matrices.

    Parameters:
    quaternions (torch.Tensor): Tensor of shape (N, 4) representing (w, x, y, z) of the quaternions.

    Returns:
    torch.Tensor: A tensor of shape (N, 3, 3) containing the rotation matrices.
    """
    # Extract the individual elements of the quaternion
    w, x, y, z = (
        quaternions[:, 0],
        quaternions[:, 1],
        quaternions[:, 2],
        quaternions[:, 3],
    )

    # Compute the rotation matrix elements
    R = torch.zeros((quaternions.size(0), 3, 3), device=quaternions.device)
    R[:, 0, 0] = 1 - 2 * (y**2 + z**2)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x**2 + z**2)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x**2 + y**2)

    return R


def rotmat2quat(R: torch.Tensor) -> torch.Tensor:
    """
    Convert a 3x3 rotation matrix into a normalized quaternion.
    Quaternion format: [w, x, y, z]
    
    Args:
        R (torch.Tensor): 3x3 rotation matrix (dtype: float, device: cpu or cuda).
    
    Returns:
        torch.Tensor: Normalized quaternion [w, x, y, z].
    """
    assert R.shape == (3, 3), "Input must be a 3x3 matrix"

    trace = torch.trace(R)
    if trace > 0:
        s = 2.0 * torch.sqrt(trace + 1.0)
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            s = 2.0 * torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

    q = torch.tensor([w, x, y, z], dtype=R.dtype, device=R.device)
    q = q / torch.norm(q)  # Normalize
    return q