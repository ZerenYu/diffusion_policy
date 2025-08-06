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
from typing import Optional, Sequence, Dict, Any
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.real_world.video_recorder import read_video
from diffusion_policy.common.cv2_util import get_image_transform
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
register_codecs()

import lerobot

# LeRobot imports
# try:
#     from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
#     from lerobot import LeRobotDataset
#     import tyro
#     LEROBOT_AVAILABLE = True
# except ImportError:
#     print("Warning: LeRobot not available. Install with: pip install lerobot")
#     LEROBOT_AVAILABLE = False


def convert_dp_to_lerobot(
    dataset_path: str,
    output_dir: Optional[str] = None,
    robot_type: str = "panda",
    fps: int = 10,
    image_keys: Optional[Sequence[str]] = None,
    state_keys: Optional[Sequence[str]] = None,
    action_key: str = "action",
    task_key: Optional[str] = None,
    push_to_hub: bool = False,
    repo_name: str = "diffusion_policy_dataset"
):
    """
    Convert Diffusion Policy data to LeRobot format.
    
    Args:
        dataset_path: Path to Diffusion Policy dataset
        output_dir: Output directory (if None, uses HF_LEROBOT_HOME)
        robot_type: Robot type for LeRobot dataset
        fps: Frames per second for the dataset
        image_keys: Camera keys to include (e.g., ['camera_0', 'camera_1'])
        state_keys: Keys to include as state/proprioceptive data
        action_key: Key for action data
        task_key: Key for task/instruction data (if available)
        push_to_hub: Whether to push to Hugging Face Hub
        repo_name: Name for the LeRobot dataset repository
    """
    
    # if not LEROBOT_AVAILABLE:
    #     raise ImportError("LeRobot is required for this conversion. Install with: pip install lerobot")
    
    # Verify input path
    input_path = pathlib.Path(os.path.expanduser(dataset_path))
    in_zarr_path = input_path.joinpath('replay_buffer.zarr')
    in_video_dir = input_path.joinpath('videos')
    
    print(f"Converting Diffusion Policy dataset:")
    print(f"  Input path: {dataset_path}")
    print(f"  Zarr path: {in_zarr_path}")
    print(f"  Video directory: {in_video_dir}")
    
    # Check if paths exist
    if not in_zarr_path.is_dir():
        raise FileNotFoundError(f"Zarr directory not found: {in_zarr_path}")
    if not in_video_dir.is_dir():
        raise FileNotFoundError(f"Video directory not found: {in_video_dir}")
    
    # Load replay buffer
    print("\n=== Loading Replay Buffer ===")
    replay_buffer = ReplayBuffer.create_from_path(str(in_zarr_path.absolute()), mode='r')
    print(f"Replay buffer loaded: {replay_buffer}")
    print(f"Number of steps: {replay_buffer.n_steps}")
    print(f"Number of episodes: {len(replay_buffer.episode_lengths)}")
    
    # Calculate episode starts
    episode_starts = replay_buffer.episode_ends[:] - replay_buffer.episode_lengths[:]
    
    # Determine available keys
    available_keys = list(replay_buffer.data.keys())
    print(f"Available data keys: {available_keys}")
    
    # Set default keys if not provided
    if image_keys is None:
        # Try to detect camera keys
        image_keys = [key for key in available_keys if key.startswith('camera_')]
        if not image_keys:
            print("Warning: No camera keys found, will try to read from video files")
            image_keys = []
    
    if state_keys is None:
        # Common state keys in DP datasets
        potential_state_keys = ['obs', 'state', 'proprio', 'joint_pos', 'joint_vel']
        state_keys = [key for key in available_keys if key in potential_state_keys]
        if not state_keys:
            print("Warning: No state keys found")
            state_keys = []
    
    print(f"Using image keys: {image_keys}")
    print(f"Using state keys: {state_keys}")
    print(f"Using action key: {action_key}")
    
    # Determine image shape and features
    features = {}
    
    # Add image features
    if image_keys:
        # Try to get image shape from first episode
        for camera_key in image_keys:
            if camera_key in replay_buffer.data:
                # Get shape from zarr data if available
                img_data = replay_buffer[camera_key]
                if len(img_data.shape) == 4:  # (steps, height, width, channels)
                    height, width, channels = img_data.shape[1:]
                    features[camera_key] = {
                        "dtype": "image",
                        "shape": (height, width, channels),
                        "names": ["height", "width", "channel"],
                    }
                else:
                    # Default shape if not available
                    features[camera_key] = {
                        "dtype": "image",
                        "shape": (256, 256, 3),
                        "names": ["height", "width", "channel"],
                    }
            else:
                # Default shape for video-based images
                features[camera_key] = {
                    "dtype": "image",
                    "shape": (256, 256, 3),
                    "names": ["height", "width", "channel"],
                }
    
    # Add state features
    for state_key in state_keys:
        if state_key in replay_buffer.data:
            state_data = replay_buffer[state_key]
            if len(state_data.shape) == 2:  # (steps, state_dim)
                state_dim = state_data.shape[1]
                features[state_key] = {
                    "dtype": "float32",
                    "shape": (state_dim,),
                    "names": [state_key],
                }
            else:
                print(f"Warning: Unexpected shape for state key {state_key}: {state_data.shape}")
    
    # Add action feature
    if action_key in replay_buffer.data:
        action_data = replay_buffer[action_key]
        if len(action_data.shape) == 2:  # (steps, action_dim)
            action_dim = action_data.shape[1]
            features["actions"] = {
                "dtype": "float32",
                "shape": (action_dim,),
                "names": ["actions"],
            }
        else:
            print(f"Warning: Unexpected shape for action key {action_key}: {action_data.shape}")
    
    # Add task feature if available
    if task_key and task_key in replay_buffer.data:
        features["task"] = {
            "dtype": "string",
            "shape": (),
            "names": ["task"],
        }
    
    print(f"LeRobot features: {list(features.keys())}")
    
    # Set up output path
    if output_dir is None:
        output_path = HF_LEROBOT_HOME / repo_name
    else:
        output_path = pathlib.Path(output_dir) / repo_name
    
    # Clean up any existing dataset
    if output_path.exists():
        shutil.rmtree(output_path)
    
    # Create LeRobot dataset
    print(f"\n=== Creating LeRobot Dataset ===")
    print(f"Output path: {output_path}")
    print(f"Robot type: {robot_type}")
    print(f"FPS: {fps}")
    
    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type=robot_type,
        fps=fps,
        features=features,
        image_writer_threads=10,
        image_writer_processes=5,
    )
    
    # Get timestamps for video reading
    timestamps = replay_buffer['timestamp'][:]
    dt = timestamps[1] - timestamps[0]
    print(f"Timestamp dt: {dt}")
    
    # Process each episode
    print(f"\n=== Converting Episodes ===")
    for episode_idx, episode_length in enumerate(replay_buffer.episode_lengths):
        episode_start = episode_starts[episode_idx]
        episode_video_dir = in_video_dir.joinpath(str(episode_idx))
        
        print(f"Processing episode {episode_idx} (length: {episode_length})")
        
        if not episode_video_dir.is_dir():
            print(f"Warning: Video directory not found for episode {episode_idx}, skipping")
            continue
        
        # Get video files for this episode
        episode_video_paths = sorted(episode_video_dir.glob('*.mp4'), key=lambda x: int(x.stem))
        print(f"  Found {len(episode_video_paths)} video files")
        
        # Process each step in the episode
        for step_idx in range(episode_length):
            global_idx = episode_start + step_idx
            
            # Prepare frame data
            frame_data = {}
            
            # Add state data
            for state_key in state_keys:
                if state_key in replay_buffer.data:
                    frame_data[state_key] = replay_buffer[state_key][global_idx]
            
            # Add action data
            if action_key in replay_buffer.data:
                frame_data["actions"] = replay_buffer[action_key][global_idx]
            
            # Add task data
            if task_key and task_key in replay_buffer.data:
                task_data = replay_buffer[task_key][global_idx]
                if isinstance(task_data, bytes):
                    frame_data["task"] = task_data.decode('utf-8')
                else:
                    frame_data["task"] = str(task_data)
            
            # Add image data from videos
            for video_path in episode_video_paths:
                camera_idx = int(video_path.stem)
                camera_key = f'camera_{camera_idx}'
                
                if camera_key not in image_keys:
                    continue
                
                # Read frame from video
                try:
                    # Get video resolution
                    with av.open(str(video_path.absolute())) as container:
                        video = container.streams.video[0]
                        vcc = video.codec_context
                        resolution = (vcc.width, vcc.height)
                    
                    # Create image transform
                    image_tf = get_image_transform(
                        input_res=resolution,
                        output_res=resolution,
                        bgr_to_rgb=False
                    )
                    
                    # Read specific frame
                    frame_generator = read_video(
                        video_path=str(video_path),
                        dt=dt,
                        img_transform=image_tf,
                        thread_type='FRAME',
                        thread_count=1
                    )
                    
                    # Get the specific frame
                    for i, frame in enumerate(frame_generator):
                        if i == step_idx:
                            frame_data[camera_key] = frame
                            break
                        if i > step_idx:
                            break
                    
                except Exception as e:
                    print(f"    Warning: Failed to read frame {step_idx} from {video_path.name}: {e}")
                    continue
            
            # Add frame to dataset
            if frame_data:
                dataset.add_frame(frame_data)
            
            if step_idx % 10 == 0:
                print(f"    Processed step {step_idx}/{episode_length}")
        
        # Save episode
        dataset.save_episode()
        print(f"  Saved episode {episode_idx}")
    
    print(f"\n=== Conversion Complete ===")
    print(f"Dataset saved to: {output_path}")
    
    # Optionally push to Hugging Face Hub
    if push_to_hub:
        print(f"Pushing to Hugging Face Hub...")
        dataset.push_to_hub(
            tags=["diffusion-policy", robot_type, "converted"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )
        print(f"Successfully pushed to Hub!")
    
    return dataset


def main():
    """Main function for command line usage."""
    if len(sys.argv) < 2:
        print("Usage: python convert_to_lerobot.py <dataset_path> [--output_dir <output_dir>] [--push_to_hub]")
        print("Example: python convert_to_lerobot.py /path/to/dp_dataset --output_dir ./output --push_to_hub")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    
    # Parse additional arguments
    output_dir = None
    push_to_hub = False
    
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--output_dir" and i + 1 < len(sys.argv):
            output_dir = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--push_to_hub":
            push_to_hub = True
            i += 1
        else:
            i += 1
    
    try:
        convert_dp_to_lerobot(
            dataset_path=dataset_path,
            output_dir=output_dir,
            push_to_hub=push_to_hub
        )
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
