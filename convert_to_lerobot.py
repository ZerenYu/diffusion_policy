#!/usr/bin/env python3
"""
Convert Diffusion Policy data format to LeRobot format.

Usage: python convert_to_lerobot.py <dataset_path> [--output_dir <output_dir>] [--push_to_hub]

This script reads Diffusion Policy data (zarr + videos) and converts it to LeRobot format.
The Diffusion Policy data structure is:
- replay_buffer.zarr: Contains lowdim data (actions, observations, timestamps, etc.)
- videos/: Contains episode subdirectories with camera MP4 files

The LeRobot format will include:
- Images from cameras
- State/proprioceptive data
- Actions
- Episode metadata
"""

import sys
import os
import pathlib
import numpy as np
import zarr
import av
import cv2
import shutil
from typing import Optional, Sequence, Dict, Any, List, Tuple
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.real_world.video_recorder import read_video
from diffusion_policy.common.cv2_util import get_image_transform
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
import tqdm
register_codecs()

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../datasets/external/lerobot/src'))

from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset


class DiffusionPolicyDataset:
    """
    A class to read and organize Diffusion Policy data by episodes.
    
    This class provides easy access to episode-based data including:
    - Low-dimensional data (actions, observations, timestamps, etc.)
    - Camera images for each episode
    - Episode metadata and organization
    """
    
    def __init__(self, dataset_path: str, 
                 lowdim_keys: Optional[Sequence[str]] = None,
                 image_keys: Optional[Sequence[str]] = None):
        """
        Initialize the Diffusion Policy dataset reader.
        
        Args:
            dataset_path: Path to the dataset directory
            lowdim_keys: Keys to read from lowdim data (e.g., ['action', 'obs', 'timestamp'])
            image_keys: Camera keys to read (e.g., ['camera_0', 'camera_1'])
        """
        self.dataset_path = pathlib.Path(os.path.expanduser(dataset_path))
        self.lowdim_keys = lowdim_keys
        self.image_keys = image_keys
        
        # Verify paths
        self.zarr_path = self.dataset_path.joinpath('replay_buffer.zarr')
        self.video_dir = self.dataset_path.joinpath('videos')
        
        if not self.zarr_path.is_dir():
            raise FileNotFoundError(f"Zarr directory not found: {self.zarr_path}")
        if not self.video_dir.is_dir():
            raise FileNotFoundError(f"Video directory not found: {self.video_dir}")
        
        # Load replay buffer
        self.replay_buffer = ReplayBuffer.create_from_path(str(self.zarr_path.absolute()), mode='r')
        
        # Calculate episode boundaries
        self.episode_lengths = self.replay_buffer.episode_lengths[:]
        self.episode_ends = self.replay_buffer.episode_ends[:]
        self.episode_starts = self.episode_ends[:] - self.episode_lengths[:]
        self.n_episodes = len(self.episode_lengths)
        
        # Set default keys if not provided
        if self.lowdim_keys is None:
            self.lowdim_keys = list(self.replay_buffer.data.keys())
        
        # Cache for video readers
        self._video_readers = {}
        self._image_transforms = {}
        
        print(f"Loaded dataset with {self.n_episodes} episodes")
        print(f"Episode lengths: {self.episode_lengths}")
        print(f"Available lowdim keys: {self.lowdim_keys}")
    
    def get_episode_info(self, episode_idx: int) -> Dict[str, Any]:
        """
        Get information about a specific episode.
        
        Args:
            episode_idx: Index of the episode
            
        Returns:
            Dictionary containing episode information
        """
        if episode_idx >= self.n_episodes:
            raise ValueError(f"Episode {episode_idx} does not exist. Total episodes: {self.n_episodes}")
        
        episode_start = self.episode_starts[episode_idx]
        episode_length = self.episode_lengths[episode_idx]
        episode_end = self.episode_ends[episode_idx]
        
        return {
            'episode_idx': episode_idx,
            'start_step': episode_start,
            'end_step': episode_end,
            'length': episode_length,
            'video_dir': self.video_dir.joinpath(str(episode_idx))
        }
    
    def get_lowdim_data(self, episode_idx: int, keys: Optional[Sequence[str]] = None) -> Dict[str, np.ndarray]:
        """
        Get low-dimensional data for a specific episode.
        
        Args:
            episode_idx: Index of the episode
            keys: Specific keys to retrieve (if None, uses self.lowdim_keys)
            
        Returns:
            Dictionary mapping keys to numpy arrays
        """
        episode_info = self.get_episode_info(episode_idx)
        start_step = episode_info['start_step']
        end_step = episode_info['end_step']
        
        if keys is None:
            keys = self.lowdim_keys
        
        data = {}
        for key in keys:
            if key in self.replay_buffer.data:
                data[key] = self.replay_buffer[key][start_step:end_step]
            else:
                print(f"Warning: Key '{key}' not found in replay buffer")
        return data
    
    def get_frame(self, episode_idx: int, step_idx: int, camera_key: str) -> np.ndarray:
        """
        Get a specific frame from a camera for a given episode and step.
        
        Args:
            episode_idx: Index of the episode
            step_idx: Index of the step within the episode
            camera_key: Camera key (e.g., 'camera_0')
            
        Returns:
            Frame as numpy array
        """
        episode_info = self.get_episode_info(episode_idx)
        video_dir = episode_info['video_dir']
        
        if not video_dir.is_dir():
            raise FileNotFoundError(f"Video directory not found for episode {episode_idx}")
        
        # Extract camera index from camera_key
        if not camera_key.startswith('camera_'):
            raise ValueError(f"Invalid camera key: {camera_key}. Expected format: 'camera_X'")
        
        camera_idx = int(camera_key.split('_')[1])
        video_path = video_dir.joinpath(f"{camera_idx}.mp4")
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Get or create video reader
        reader_key = f"{episode_idx}_{camera_idx}"
        if reader_key not in self._video_readers:
            # Get video properties
            with av.open(str(video_path)) as container:
                video = container.streams.video[0]
                vcc = video.codec_context
                resolution = (vcc.width, vcc.height)
                fps = video.average_rate
                duration = video.duration
            
            # Get timestamps for this episode
            episode_data = self.get_lowdim_data(episode_idx, ['timestamp'])
            if 'timestamp' in episode_data:
                timestamps = episode_data['timestamp']
                dt = timestamps[1] - timestamps[0] if len(timestamps) > 1 else 1.0
            else:
                dt = 1.0 / float(fps) if fps else 1.0
            
            # Create image transform
            image_tf = get_image_transform(
                input_res=resolution,
                output_res=resolution,
                bgr_to_rgb=False
            )
            
            self._video_readers[reader_key] = {
                'video_path': str(video_path),
                'dt': dt,
                'image_tf': image_tf,
                'resolution': resolution,
                'fps': fps
            }
        
        reader_info = self._video_readers[reader_key]
        
        # Read the specific frame
        frame_generator = read_video(
            video_path=reader_info['video_path'],
            dt=reader_info['dt'],
            img_transform=reader_info['image_tf'],
            thread_type='FRAME',
            thread_count=1
        )
        
        # Skip to the desired frame
        for i, frame in enumerate(frame_generator):
            if i == step_idx:
                return frame
            elif i > step_idx:
                break
        
        raise ValueError(f"Step {step_idx} not found in episode {episode_idx}, camera {camera_key}")
    
    def get_frame_data(self, episode_idx: int, step_idx: int) -> Dict[str, Any]:
        """
        Get all data for a single frame including low-dimensional data and all camera frames.
        
        Args:
            episode_idx: Index of the episode
            step_idx: Index of the step within the episode
            
        Returns:
            Dictionary containing all frame data:
            - episode_info: Episode metadata
            - step_idx: Step index within episode
            - lowdim_data: Dictionary of low-dimensional data for this step
            - frames: Dictionary mapping camera keys to frame arrays
            - available_cameras: List of available cameras
        """
        episode_info = self.get_episode_info(episode_idx)
        
        if step_idx >= episode_info['length']:
            raise ValueError(f"Step {step_idx} is out of range for episode {episode_idx} (length: {episode_info['length']})")
        
        # Get low-dimensional data for this step
        episode_lowdim = self.get_lowdim_data(episode_idx)
        lowdim_data = {}
        for key, data in episode_lowdim.items():
            if step_idx < len(data):
                lowdim_data[key] = data[step_idx]
            else:
                print(f"Warning: Step {step_idx} not available for key '{key}' (data length: {len(data)})")
        
        # Get available cameras
        available_cameras = self.get_available_cameras(episode_idx)
        
        # Get frames from all available cameras
        frames = {}
        for camera_key in available_cameras:
            try:
                frame = self.get_frame(episode_idx, step_idx, camera_key)
                frames[camera_key] = frame
            except Exception as e:
                print(f"Warning: Could not read frame from {camera_key} at step {step_idx}: {e}")
        return {
            'episode_info': episode_info,
            'step_idx': step_idx,
            'lowdim_data': lowdim_data,
            'frames': frames,
            'available_cameras': available_cameras,
            'timestamp': lowdim_data.get('timestamp', None),
            'action': lowdim_data.get('action', None).astype(np.float32),
            'stage': lowdim_data.get('stage', None),
            'robot_eef_pose': lowdim_data.get('robot_eef_pose', None).astype(np.float32),
            'robot_eef_pose_vel': lowdim_data.get('robot_eef_pose_vel', None).astype(np.float32),
            'robot_joint': lowdim_data.get('robot_joint', None).astype(np.float32),
            'robot_joint_vel': lowdim_data.get('robot_joint_vel', None).astype(np.float32)
        }
    
    def get_features(self, episode_idx: int, step_idx: int, task: str = "manipulation") -> Dict[str, Any]:
        """
        Get formatted features for LeRobot dataset from a single frame.
        
        This function extracts and formats all data for a single frame in the format
        expected by LeRobot's add_frame() method.
        
        Args:
            episode_idx: Index of the episode
            step_idx: Index of the step within the episode
            task: Task name for this frame (default: "manipulation")
            
        Returns:
            Dictionary containing formatted features for LeRobot dataset:
            - image: Main camera image (if available)
            - wrist_image: Wrist camera image (if available)
            - state: Robot state/proprioceptive data
            - actions: Robot actions
            - timestamp: Frame timestamp
        """
        frame_data = self.get_frame_data(episode_idx, step_idx)
        # Initialize features dictionary
        features = {}
        
        # Add images
        frames = frame_data['frames']
        if 'camera_0' in frames:
            features['image'] = frames['camera_0']
        if 'camera_1' in frames:
            features['wrist_image'] = frames['camera_1']
        
        # Add state data (combine robot proprioceptive information)
        state_components = []
        lowdim_data = frame_data['lowdim_data']
        # Add robot joint positions if available
        if 'robot_joint' in frame_data:
            joint_pos = frame_data['robot_joint']
            if isinstance(joint_pos, np.ndarray):
                features['robot_joint'] = joint_pos.flatten()
        
        # Add robot joint velocities if available
        if 'robot_joint_vel' in frame_data:
            joint_vel = frame_data['robot_joint_vel']
            if isinstance(joint_vel, np.ndarray):
                features['robot_joint_vel'] = joint_vel.flatten()
        
        # Add end-effector pose if available
        if 'robot_eef_pose' in frame_data:
            eef_pose = frame_data['robot_eef_pose']
            if isinstance(eef_pose, np.ndarray):
                features['robot_eef_pose'] = eef_pose.flatten()
        
        # Add end-effector velocity if available
        if 'robot_eef_pose_vel' in frame_data:
            eef_vel = frame_data['robot_eef_pose_vel']
            if isinstance(eef_vel, np.ndarray):
                features['robot_eef_pose_vel'] = eef_vel.flatten()
        
        # Add actions
        if 'action' in lowdim_data and lowdim_data['action'] is not None:
            action = lowdim_data['action']
            if isinstance(action, np.ndarray):
                features['actions'] = action.flatten()
        
        # Add timestamp
        if 'timestamp' in lowdim_data and lowdim_data['timestamp'] is not None:
            timestamp = lowdim_data['timestamp']
            if isinstance(timestamp, (np.ndarray, list)):
                # Extract scalar value if it's an array
                if hasattr(timestamp, '__len__') and len(timestamp) > 0:
                    features['timestamp'] = float(timestamp[0] if isinstance(timestamp, np.ndarray) else timestamp[0])
                else:
                    features['timestamp'] = float(timestamp)
            else:
                features['timestamp'] = float(timestamp)
        return features
    
    def get_lerobot_features_spec(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the features specification for LeRobot dataset creation.
        
        This function analyzes the dataset and returns a features dictionary
        that can be used to create a LeRobot dataset.
        
        Returns:
            Dictionary containing feature specifications for LeRobot dataset
        """
        # Analyze first episode to determine data shapes and types
        if self.n_episodes == 0:
            raise ValueError("No episodes available in dataset")
        
        # Get sample frame data
        sample_frame = self.get_frame_data(0, 0)
        features = self.get_features(0, 0)
        # Build features specification
        features_spec = {}
        # Add image features
        if 'image' in features:
            image_shape = features['image'].shape
            features_spec['image'] = {
                'dtype': 'image',
                'shape': image_shape,
                'names': ['height', 'width', 'channel'] if len(image_shape) == 3 else ['height', 'width']
            }
        
        if 'wrist_image' in features:
            wrist_shape = features['wrist_image'].shape
            features_spec['wrist_image'] = {
                'dtype': 'image',
                'shape': wrist_shape,
                'names': ['height', 'width', 'channel'] if len(wrist_shape) == 3 else ['height', 'width']
            }
        
        # Add state feature
        if 'state' in features:
            state_shape = features['state'].shape
            features_spec['state'] = {
                'dtype': 'float32',
                'shape': state_shape,
                'names': ['state']
            }
        
        # Add actions feature
        if 'actions' in features:
            action_shape = features['actions'].shape
            features_spec['actions'] = {
                'dtype': 'float32',
                'shape': action_shape,
                'names': ['actions']
            }
            
        if 'robot_joint' in features:
            robot_joint_shape = features['robot_joint'].shape
            features_spec['robot_joint'] = {
                'dtype': 'float32',
                'shape': robot_joint_shape,
                'names': ['robot_joint']
            }
            
        if 'robot_joint_vel' in features:
            robot_joint_vel_shape = features['robot_joint_vel'].shape
            features_spec['robot_joint_vel'] = {
                'dtype': 'float32',
                'shape': robot_joint_vel_shape,
                'names': ['robot_joint_vel']
            }
            
        if 'robot_eef_pose' in features:
            robot_eef_pose_shape = features['robot_eef_pose'].shape
            features_spec['robot_eef_pose'] = {
                'dtype': 'float32',
                'shape': robot_eef_pose_shape,
                'names': ['robot_eef_pose']
            }
            
        if 'robot_eef_pose_vel' in features:
            robot_eef_pose_vel_shape = features['robot_eef_pose_vel'].shape
            features_spec['robot_eef_pose_vel'] = {
                'dtype': 'float32',
                'shape': robot_eef_pose_vel_shape,
                'names': ['robot_eef_pose_vel']
            }
            
        if 'stage' in features:
            stage_shape = features['stage'].shape
            features_spec['stage'] = {
                'dtype': 'string',
                'shape': stage_shape,
                'names': ['stage']
            }
            
        
        return features_spec
    
    def get_episode_frames(self, episode_idx: int, camera_keys: Optional[Sequence[str]] = None) -> Dict[str, List[np.ndarray]]:
        """
        Get all frames for an episode from specified cameras.
        
        Args:
            episode_idx: Index of the episode
            camera_keys: List of camera keys to read (if None, uses self.image_keys)
            
        Returns:
            Dictionary mapping camera keys to lists of frames
        """
        episode_info = self.get_episode_info(episode_idx)
        episode_length = episode_info['length']
        
        if camera_keys is None:
            camera_keys = self.image_keys or []
        
        frames = {}
        for camera_key in camera_keys:
            frames[camera_key] = []
            for step_idx in range(episode_length):
                try:
                    frame = self.get_frame(episode_idx, step_idx, camera_key)
                    frames[camera_key].append(frame)
                    break
                except Exception as e:
                    print(f"Warning: Could not read frame {step_idx} for {camera_key}: {e}")
                    break
        
        return frames
    
    def get_available_cameras(self, episode_idx: int) -> List[str]:
        """
        Get list of available cameras for an episode.
        
        Args:
            episode_idx: Index of the episode
            
        Returns:
            List of available camera keys
        """
        episode_info = self.get_episode_info(episode_idx)
        video_dir = episode_info['video_dir']
        
        if not video_dir.is_dir():
            return []
        
        camera_keys = []
        for video_file in video_dir.glob('*.mp4'):
            try:
                camera_idx = int(video_file.stem)
                camera_keys.append(f'camera_{camera_idx}')
            except ValueError:
                continue
        
        return sorted(camera_keys)
    
    def get_dataset_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the dataset.
        
        Returns:
            Dictionary containing dataset summary information
        """
        summary = {
            'n_episodes': self.n_episodes,
            'episode_lengths': self.episode_lengths.tolist(),
            'total_steps': sum(self.episode_lengths),
            'available_lowdim_keys': self.lowdim_keys,
            'episode_info': []
        }
        
        for episode_idx in range(self.n_episodes):
            episode_info = self.get_episode_info(episode_idx)
            available_cameras = self.get_available_cameras(episode_idx)
            episode_info['available_cameras'] = available_cameras
            summary['episode_info'].append(episode_info)
        
        return summary
    
    def __len__(self) -> int:
        """Return the number of episodes in the dataset."""
        return self.n_episodes
    
    def __getitem__(self, episode_idx: int) -> Dict[str, Any]:
        """
        Get all data for an episode.
        
        Args:
            episode_idx: Index of the episode
            
        Returns:
            Dictionary containing episode data
        """
        lowdim_data = self.get_lowdim_data(episode_idx)
        frames = self.get_episode_frames(episode_idx)
        episode_info = self.get_episode_info(episode_idx)
        
        return {
            'episode_info': episode_info,
            'lowdim_data': lowdim_data,
            'frames': frames
        }

def demonstrate_dataset_usage(dataset_path: str):
    """
    Demonstrate how to use the DiffusionPolicyDataset class.
    
    Args:
        dataset_path: Path to the Diffusion Policy dataset
    """
    print("=" * 60)
    print("DEMONSTRATING DIFFUSION POLICY DATASET USAGE")
    print("=" * 60)
    
    # Initialize the dataset
    dataset = DiffusionPolicyDataset(
        dataset_path=dataset_path,
        lowdim_keys=['action', 'obs', 'timestamp', 'robot_eef_pose',
                     'robot_eef_pose_vel', 'robot_joint', 'robot_joint_vel','stage'],
        image_keys=['camera_0']
    )
    
    # Get dataset summary
    summary = dataset.get_dataset_summary()
    print(f"\nDataset Summary:")
    print(f"  Total episodes: {summary['n_episodes']}")
    print(f"  Total steps: {summary['total_steps']}")
    print(f"  Episode lengths: {summary['episode_lengths']}")
    
    # Process first episode as an example
    if len(dataset) > 0:
        episode_idx = 0
        print(f"\n--- Processing Episode {episode_idx} ---")
        
        # Get episode info
        episode_info = dataset.get_episode_info(episode_idx)
        print(f"Episode info: {episode_info}")
        
        # Get available cameras
        available_cameras = dataset.get_available_cameras(episode_idx)
        print(f"Available cameras: {available_cameras}")
        
        # Get lowdim data
        lowdim_data = dataset.get_lowdim_data(episode_idx)
        print(f"Lowdim data keys: {list(lowdim_data.keys())}")
        for key, data in lowdim_data.items():
            print(f"  {key}: shape={data.shape}, dtype={data.dtype}")
        
        # Get a specific frame
        if available_cameras and episode_info['length'] > 0:
            camera_key = available_cameras[0]
            step_idx = 0
            try:
                frame = dataset.get_frame(episode_idx, step_idx, camera_key)
                print(f"Frame from {camera_key} at step {step_idx}: shape={frame.shape}, dtype={frame.dtype}")
                print(f"  Frame range: [{frame.min()}, {frame.max()}]")
            except Exception as e:
                print(f"Error reading frame: {e}")
        
        # Get complete frame data (everything for a single frame)
        if episode_info['length'] > 0:
            step_idx = 0
            try:
                frame_data = dataset.get_frame_data(episode_idx, step_idx)
                print(f"\nComplete frame data for episode {episode_idx}, step {step_idx}:")
                print(f"  Episode info: {frame_data['episode_info']}")
                print(f"  Available cameras: {frame_data['available_cameras']}")
                print(f"  Lowdim data keys: {list(frame_data['lowdim_data'].keys())}")
                print(f"  Frame keys: {list(frame_data['frames'].keys())}")
                print(f'frame_data: {frame_data}')
                
                # Show some data details
                for key, value in frame_data['lowdim_data'].items():
                    if isinstance(value, np.ndarray):
                        print(f"    {key}: shape={value.shape}, dtype={value.dtype}")
                    else:
                        print(f"    {key}: {value}")
                
                for camera_key, frame in frame_data['frames'].items():
                    print(f"    {camera_key}: shape={frame.shape}, dtype={frame.dtype}")
                    print(f"      Frame range: [{frame.min()}, {frame.max()}]")
                    
            except Exception as e:
                print(f"Error reading complete frame data: {e}")
        
        # Test get_features function
        if episode_info['length'] > 0:
            step_idx = 0
            try:
                features = dataset.get_features(episode_idx, step_idx)
                print(f"\nLeRobot features for episode {episode_idx}, step {step_idx}:")
                print(f"  Feature keys: {list(features.keys())}")
                for key, value in features.items():
                    if isinstance(value, np.ndarray):
                        print(f"    {key}: shape={value.shape}, dtype={value.dtype}")
                        if key in ['image', 'wrist_image']:
                            print(f"      Range: [{value.min()}, {value.max()}]")
                    else:
                        print(f"    {key}: {value}")
                
                # Test features specification
                features_spec = dataset.get_lerobot_features_spec()
                print(f"\nLeRobot features specification:")
                for key, spec in features_spec.items():
                    print(f"  {key}: {spec}")
                    
            except Exception as e:
                print(f"Error getting LeRobot features: {e}")
        
        # Get all frames for the episode
        try:
            frames = dataset.get_episode_frames(episode_idx, available_cameras[:1])
            print(f"Retrieved frames for cameras: {list(frames.keys())}")
            
            for camera_key, frame_list in frames.items():
                print(f"  {camera_key}: {len(frame_list)} frames")
        except Exception as e:
            print(f"Error reading episode frames: {e}")
    
    print("\n" + "=" * 60)

def convert_to_lerobot(dataset_path: str, output_path: str, robot_type: str = "UR10", fps: int = 10):
    """
    Convert a Diffusion Policy dataset to LeRobot format.
    
    Args:
        dataset_path: Path to the Diffusion Policy dataset
    """
    dp_dataset = DiffusionPolicyDataset(
        dataset_path=dataset_path,
        lowdim_keys=['action', 'timestamp', 'robot_eef_pose',
                     'robot_eef_pose_vel', 'robot_joint', 'robot_joint_vel','stage'],
        image_keys=['camera_0']
    )
    features = dp_dataset.get_lerobot_features_spec()
    print(f'features: {features}')
    Le_dataset = LeRobotDataset.create(
        repo_id=dataset_path.split('/')[-1],
        robot_type=robot_type,
        fps=fps,
        features=features,
        image_writer_threads=20,
        image_writer_processes=10,
    )
    for episode_idx in range(dp_dataset.n_episodes):
        episode_data = dp_dataset[episode_idx]
        for step_idx in tqdm.trange(episode_data['episode_info']['length']):
            frame_data = dp_dataset.get_frame_data(episode_idx, step_idx)
            Le_dataset.add_frame({
                'image': frame_data['frames']['camera_0'],
                'robot_joint': frame_data['robot_joint'],
                'robot_joint_vel': frame_data['robot_joint_vel'],
                'robot_eef_pose': frame_data['robot_eef_pose'],
                'robot_eef_pose_vel': frame_data['robot_eef_pose_vel'],
                'actions': frame_data['action'],
                }, task='pusht', timestamp=frame_data['timestamp'])
        Le_dataset.save_episode()
    

if __name__ == "__main__":
    dataset_path = "data/experiments/test_push_T_2"
    output_path = "data/experiments/test_push_T_2_lerobot"
    convert_to_lerobot(dataset_path, output_path)

