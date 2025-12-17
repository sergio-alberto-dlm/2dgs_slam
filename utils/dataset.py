import glob
import os
import cv2
import numpy as np
import torch
import trimesh
from gaussian_splatting.utils.graphics_utils import focal2fov


class ReplicaParser:
    def __init__(self, input_folder):
        self.input_folder = input_folder
        self.color_paths = sorted(glob.glob(f"{self.input_folder}/results/frame*.jpg"))
        self.depth_paths = sorted(glob.glob(f"{self.input_folder}/results/depth*.png"))
        self.n_img = len(self.color_paths)
        self.load_poses(f"{self.input_folder}/traj.txt")

    def load_poses(self, path):
        self.poses = []
        with open(path, "r") as f:
            lines = f.readlines()

        frames = []
        for i in range(self.n_img):
            line = lines[i]
            pose = np.array(list(map(float, line.split()))).reshape(4, 4)
            pose = np.linalg.inv(pose)
            self.poses.append(pose)
            frame = {
                "file_path": self.color_paths[i],
                "depth_path": self.depth_paths[i],
                "transform_matrix": pose.tolist(),
            }

            frames.append(frame)
        self.frames = frames


class TUMParser:
    def __init__(self, input_folder):
        self.input_folder = input_folder
        self.load_poses(self.input_folder, frame_rate=32)
        self.n_img = len(self.color_paths)

    def parse_list(self, filepath, skiprows=0):
        data = np.loadtxt(filepath, delimiter=" ", dtype=np.str_, skiprows=skiprows)
        return data

    def associate_frames(self, tstamp_image, tstamp_depth, tstamp_pose, max_dt=0.08):
        associations = []
        for i, t in enumerate(tstamp_image):
            if tstamp_pose is None:
                j = np.argmin(np.abs(tstamp_depth - t))
                if np.abs(tstamp_depth[j] - t) < max_dt:
                    associations.append((i, j))

            else:
                j = np.argmin(np.abs(tstamp_depth - t))
                k = np.argmin(np.abs(tstamp_pose - t))

                if (np.abs(tstamp_depth[j] - t) < max_dt) and (
                    np.abs(tstamp_pose[k] - t) < max_dt
                ):
                    associations.append((i, j, k))

        return associations

    def load_poses(self, datapath, frame_rate=-1):
        if os.path.isfile(os.path.join(datapath, "groundtruth.txt")):
            pose_list = os.path.join(datapath, "groundtruth.txt")
        elif os.path.isfile(os.path.join(datapath, "pose.txt")):
            pose_list = os.path.join(datapath, "pose.txt")

        image_list = os.path.join(datapath, "rgb.txt")
        depth_list = os.path.join(datapath, "depth.txt")

        image_data = self.parse_list(image_list)
        depth_data = self.parse_list(depth_list)
        pose_data = self.parse_list(pose_list, skiprows=1)
        pose_vecs = pose_data[:, 0:].astype(np.float64)

        tstamp_image = image_data[:, 0].astype(np.float64)
        tstamp_depth = depth_data[:, 0].astype(np.float64)
        tstamp_pose = pose_data[:, 0].astype(np.float64)
        associations = self.associate_frames(tstamp_image, tstamp_depth, tstamp_pose)

        indicies = [0]
        for i in range(1, len(associations)):
            t0 = tstamp_image[associations[indicies[-1]][0]]
            t1 = tstamp_image[associations[i][0]]
            if t1 - t0 > 1.0 / frame_rate:
                indicies += [i]

        self.color_paths, self.poses, self.depth_paths, self.frames = [], [], [], []

        for ix in indicies:
            (i, j, k) = associations[ix]
            self.color_paths += [os.path.join(datapath, image_data[i, 1])]
            self.depth_paths += [os.path.join(datapath, depth_data[j, 1])]

            quat = pose_vecs[k][4:]
            trans = pose_vecs[k][1:4]
            T = trimesh.transformations.quaternion_matrix(np.roll(quat, 1))
            T[:3, 3] = trans
            self.poses += [np.linalg.inv(T)]
            frame = {
                "file_path": str(os.path.join(datapath, image_data[i, 1])),
                "depth_path": str(os.path.join(datapath, depth_data[j, 1])),
                "transform_matrix": (np.linalg.inv(T)).tolist(),
            }
            self.frames.append(frame)


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, args, path, config):
        self.args = args
        self.path = path
        self.config = config
        self.device = "cuda:0"
        self.dtype = torch.float32
        self.num_imgs = 999999

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, idx):
        pass
    
    def update_intrinsics(self, intrinsics_matrix):
        """
        Update dataset intrinsic parameters from a 3x3 intrinsics matrix.
        This is used when monocular depth estimation provides updated intrinsics.
        
        Args:
            intrinsics_matrix: 3x3 numpy array or torch tensor with camera intrinsics
        """
        pass  # Base class does nothing


class MonocularDataset(BaseDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        calibration = config["Dataset"]["Calibration"]
        # Camera prameters
        self.fx = calibration["fx"]
        self.fy = calibration["fy"]
        self.cx = calibration["cx"]
        self.cy = calibration["cy"]
        self.width = calibration["width"]
        self.height = calibration["height"]
        self.fovx = focal2fov(self.fx, self.width)
        self.fovy = focal2fov(self.fy, self.height)
        self.K = np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )
        # distortion parameters
        self.disorted = calibration["distorted"]
        self.dist_coeffs = np.array(
            [
                calibration["k1"],
                calibration["k2"],
                calibration["p1"],
                calibration["p2"],
                calibration["k3"],
            ]
        )
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            self.K,
            self.dist_coeffs,
            np.eye(3),
            self.K,
            (self.width, self.height),
            cv2.CV_32FC1,
        )
        # depth parameters
        self.has_depth = True if "depth_scale" in calibration.keys() else False
        self.depth_scale = calibration["depth_scale"] if self.has_depth else None

        # Default scene scale
        nerf_normalization_radius = 5
        self.scene_info = {
            "nerf_normalization": {
                "radius": nerf_normalization_radius,
                "translation": np.zeros(3),
            },
        }

    def __getitem__(self, idx):
        color_path = self.color_paths[idx]
        pose = self.poses[idx]
        image = cv2.imread(color_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        depth = None

        if self.disorted:
            image = cv2.remap(image, self.map1x, self.map1y, cv2.INTER_LINEAR)

        if self.has_depth:
            depth_path = self.depth_paths[idx]
            depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH) / self.depth_scale

        image = (
            torch.from_numpy(image / 255.0)
            .clamp(0.0, 1.0)
            .permute(2, 0, 1)
            .to(device=self.device, dtype=self.dtype)
        )
        pose = torch.from_numpy(pose).to(device=self.device)

        return image, depth, pose
    
    def update_intrinsics(self, intrinsics_matrix):
        """
        Update dataset intrinsic parameters from a 3x3 intrinsics matrix.
        This is used when monocular depth estimation provides updated intrinsics.
        
        Args:
            intrinsics_matrix: 3x3 numpy array or torch tensor with camera intrinsics
        """
        import torch
        
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
        
        # Update K matrix
        self.K = np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )
        
        # Update FoV
        self.fovx = focal2fov(self.fx, self.width)
        self.fovy = focal2fov(self.fy, self.height)
        
        # Update undistortion maps if using distortion correction
        if self.disorted:
            self.map1x, self.map1y = cv2.initUndistortRectifyMap(
                self.K,
                self.dist_coeffs,
                np.eye(3),
                self.K,
                (self.width, self.height),
                cv2.CV_32FC1,
            )
        
        print(f"Updated dataset intrinsics: fx={self.fx:.2f}, fy={self.fy:.2f}, cx={self.cx:.2f}, cy={self.cy:.2f}")


class TUMDataset(MonocularDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        dataset_path = config["Dataset"]["dataset_path"]
        parser = TUMParser(dataset_path)
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.depth_paths = parser.depth_paths
        self.poses = parser.poses


class ReplicaDataset(MonocularDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        dataset_path = config["Dataset"]["dataset_path"]
        parser = ReplicaParser(dataset_path)
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.depth_paths = parser.depth_paths
        self.poses = parser.poses


class VideoParser:
    def __init__(self, video_path, frame_rate=-1, start_frame=0, end_frame=-1):
        """
        Parser for RGB video files.
        
        Args:
            video_path: Path to the video file
            frame_rate: Target frame rate for sampling (frames per second). -1 means use all frames
            start_frame: Starting frame index (default: 0)
            end_frame: Ending frame index (default: -1, meaning last frame)
        """
        self.video_path = video_path
        self.frame_rate = frame_rate
        self.start_frame = start_frame
        self.end_frame = end_frame
        
        # Open video and get properties
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Determine frame indices to use
        self.load_frame_indices()
        self.n_img = len(self.frame_indices)
        
        # Initialize poses as identity matrices (SLAM will estimate these)
        self.poses = [np.eye(4) for _ in range(self.n_img)]
        
        self.cap.release()
    
    def load_frame_indices(self):
        """Determine which frames to extract from the video."""
        end_frame = self.end_frame if self.end_frame > 0 else self.total_frames
        end_frame = min(end_frame, self.total_frames)
        
        if self.frame_rate <= 0:
            # Use all frames in range
            self.frame_indices = list(range(self.start_frame, end_frame))
        else:
            # Sample frames at specified frame rate
            frame_interval = max(1, int(self.video_fps / self.frame_rate))
            self.frame_indices = list(range(self.start_frame, end_frame, frame_interval))
    
    def get_frame(self, frame_idx):
        """Extract a specific frame from the video."""
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError(f"Could not read frame {frame_idx} from video")
        
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


class VideoDataset(MonocularDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        
        # Get video configuration
        dataset_config = config["Dataset"]
        video_path = dataset_config["dataset_path"]
        frame_rate = dataset_config.get("frame_rate", -1)
        start_frame = dataset_config.get("start_frame", 0)
        end_frame = dataset_config.get("end_frame", -1)
        
        # Get resize percentage (default: 100 = no resize)
        self.resize_percent = dataset_config.get("resize_percent", 100.0)
        
        # Initialize parser
        self.parser = VideoParser(video_path, frame_rate, start_frame, end_frame)
        self.num_imgs = self.parser.n_img
        self.poses = self.parser.poses
        
        # Update camera intrinsics and dimensions if resizing
        if self.resize_percent != 100.0:
            scale = self.resize_percent / 100.0
            self.width = int(self.parser.video_width * scale)
            self.height = int(self.parser.video_height * scale)
            self.fx *= scale
            self.fy *= scale
            self.cx *= scale
            self.cy *= scale
            
            # Update K matrix
            self.K = np.array(
                [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
            )
            
            # Update FOV
            self.fovx = focal2fov(self.fx, self.width)
            self.fovy = focal2fov(self.fy, self.height)
            
            # Update undistortion maps if using distortion correction
            if self.disorted:
                self.map1x, self.map1y = cv2.initUndistortRectifyMap(
                    self.K,
                    self.dist_coeffs,
                    np.eye(3),
                    self.K,
                    (self.width, self.height),
                    cv2.CV_32FC1,
                )
        else:
            # Use original video dimensions
            self.width = self.parser.video_width
            self.height = self.parser.video_height
        
        # No depth for video dataset
        self.has_depth = False
        self.depth_paths = [None] * self.num_imgs
    
    def __getitem__(self, idx):
        """Get a frame from the video."""
        frame_idx = self.parser.frame_indices[idx]
        pose = self.poses[idx]
        
        # Extract frame from video
        image = self.parser.get_frame(frame_idx)
        depth = None
        
        # Resize if needed
        if self.resize_percent != 100.0:
            new_size = (self.width, self.height)
            image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
        
        # Apply undistortion if needed
        if self.disorted:
            image = cv2.remap(image, self.map1x, self.map1y, cv2.INTER_LINEAR)
        
        # Convert to tensor
        image = (
            torch.from_numpy(image / 255.0)
            .clamp(0.0, 1.0)
            .permute(2, 0, 1)
            .to(device=self.device, dtype=self.dtype)
        )
        pose = torch.from_numpy(pose).to(device=self.device)
        
        return image, depth, pose


def load_dataset(args, path, config):
    if config["Dataset"]["type"] == "tum":
        return TUMDataset(args, path, config)
    elif config["Dataset"]["type"] == "replica":
        return ReplicaDataset(args, path, config)
    elif config["Dataset"]["type"] == "video":
        return VideoDataset(args, path, config)
    else:
        raise ValueError("Unknown dataset type")