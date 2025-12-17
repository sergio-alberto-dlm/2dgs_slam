import rich
import rerun as rr
from utils.normal_utils import normal_to_rgb 
from utils.pose_utils import rotmat2quat

_log_styles = {
    "2dgs_slam": "bold sandy_brown",
    "GUI": "bold magenta",
    "Eval": "bold red",
    "GP": "bold green"
}

def get_style(tag):
    if tag in _log_styles.keys():
        return _log_styles[tag]
    return "bold blue"


def Log(*args, tag="2dgs_slam"):
    style = get_style(tag)
    rich.print(f"[{style}]{tag}:[/{style}]", *args)


def log_to_rerun(
    render_pkg, 
    viewpoint, 
    poses_xyz_data, 
    sequence,  
):
    LOG_IMG_EVERY = 10
    rr.set_time("frame", sequence=sequence)

    # unpack visuals 
    ori_rgb = (viewpoint.original_image * 255).permute(1, 2, 0).cpu().numpy() 
    render_rgb = (render_pkg["render"] * 255).permute(1, 2, 0).detach().cpu().numpy()
    depth = (render_pkg["depth"]).squeeze().detach().cpu().numpy()
    normal = (render_pkg["rend_normal"]).permute(1, 2, 0).detach().cpu().numpy()
    normal = normal_to_rgb(normal)

    # unpack poses 
    q_est, t_est = rotmat2quat(viewpoint.T[:3, :3]).cpu().numpy(), viewpoint.T[:3, 3].cpu().numpy()
    poses_est_xyz, poses_gt_xyz = poses_xyz_data
 
    # log visuals 
    rr.log("vis/original", rr.Image(ori_rgb))
    rr.log("vis/normal", rr.Image(normal))
    rr.log("vis/depth", rr.DepthImage(depth))
    
    # log poses 
    rr.log(
        "world/traj", 
        rr.LineStrips3D(
            [poses_gt_xyz, poses_est_xyz], 
            colors=[[70, 230, 30], [30, 90, 230]],
            labels=["traj_gt", "traj_est"],
        )
    )

    rr.log(
        "world/cam", 
        rr.Pinhole(
            image_from_camera = viewpoint.intrins.numpy(), 
            width = viewpoint.image_width, height = viewpoint.image_height, 
        )
    )
    rr.log("world/cam", rr.Image(render_rgb))
    rr.log("world/cam", rr.Transform3D(translation=t_est, quaternion=q_est))