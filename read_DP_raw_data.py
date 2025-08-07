#!/usr/bin/env python3
"""
Test script to read raw Diffusion Policy data.
Usage: python read_DP_raw_data.py <dataset_path>
"""

import sys
import os
import pathlib
import numpy as np
import zarr
import av
import cv2
from typing import Optional, Sequence
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.real_world.video_recorder import read_video
from diffusion_policy.common.cv2_util import get_image_transform
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
register_codecs()


def read_raw_data(dataset_path: str, 
                  lowdim_keys: Optional[Sequence[str]] = None,
                  image_keys: Optional[Sequence[str]] = None,
                  max_episodes: int = 3,
                  max_steps_per_episode: int = 10,
                  save_frames: bool = False,
                  output_dir: str = "saved_frames"):
    """
    Read raw Diffusion Policy data from a dataset path.
    
    Args:
        dataset_path: Path to the dataset directory
        lowdim_keys: Keys to read from lowdim data (e.g., ['action', 'obs', 'timestamp'])
        image_keys: Camera keys to read (e.g., ['camera_0', 'camera_1'])
        max_episodes: Maximum number of episodes to process
        max_steps_per_episode: Maximum number of steps per episode to process
        save_frames: Whether to save frames to disk
        output_dir: Directory to save frames (if save_frames=True)
    """
    
    # Verify input path
    input_path = pathlib.Path(os.path.expanduser(dataset_path))
    in_zarr_path = input_path.joinpath('replay_buffer.zarr')
    in_video_dir = input_path.joinpath('videos')
    
    print(f"Dataset path: {dataset_path}")
    print(f"Zarr path: {in_zarr_path}")
    print(f"Video directory: {in_video_dir}")
    
    # Check if paths exist
    if not in_zarr_path.is_dir():
        raise FileNotFoundError(f"Zarr directory not found: {in_zarr_path}")
    if not in_video_dir.is_dir():
        raise FileNotFoundError(f"Video directory not found: {in_video_dir}")
    
    # Create output directory for saving frames
    if save_frames:
        output_path = pathlib.Path(output_dir)
        output_path.mkdir(exist_ok=True)
        print(f"Frames will be saved to: {output_path.absolute()}")
    
    # Load replay buffer
    print("\n=== Loading Replay Buffer ===")
    replay_buffer = ReplayBuffer.create_from_path(str(in_zarr_path.absolute()), mode='r')
    print(f"Replay buffer loaded: {replay_buffer}")
    print(f"Number of steps: {replay_buffer.n_steps}")
    print(f"Number of episodes: {len(replay_buffer.episode_lengths)}")
    print(f"Episode lengths: {replay_buffer.episode_lengths}")
    print(f"Episode ends: {replay_buffer.episode_ends}")
    
    # Calculate episode starts
    episode_starts = replay_buffer.episode_ends[:] - replay_buffer.episode_lengths[:]
    print(f"Episode starts: {episode_starts}")
    
    # Show available data keys
    print(f"\nAvailable data keys: {list(replay_buffer.data.keys())}")
    
    # Read lowdim data
    if lowdim_keys is None:
        lowdim_keys = list(replay_buffer.data.keys())
    
    print(f"\n=== Reading Lowdim Data ===")
    print(f"Reading keys: {lowdim_keys}")
    
    for key in lowdim_keys:
        if key in replay_buffer.data:
            data = replay_buffer[key][:]
            print(f"{key}: shape={data.shape}, dtype={data.dtype}")
            
            # Show first few values
            if len(data.shape) == 1:
                print(f"  First 5 values: {data[:5]}")
            elif len(data.shape) == 2:
                print(f"  First 3 rows: {data[:3]}")
            else:
                print(f"  First element shape: {data[0].shape}")
        else:
            print(f"Key '{key}' not found in replay buffer")
    
    # Read image data
    if image_keys is not None:
        print(f"\n=== Reading Image Data ===")
        print(f"Reading cameras: {image_keys}")
        
        # Get timestamps for video reading
        timestamps = replay_buffer['timestamp'][:]
        dt = timestamps[1] - timestamps[0]
        print(f"Timestamp dt: {dt}")
        
        # Process episodes
        for episode_idx in range(min(len(replay_buffer.episode_lengths), max_episodes)):
            episode_length = replay_buffer.episode_lengths[episode_idx]
            episode_start = episode_starts[episode_idx]
            episode_video_dir = in_video_dir.joinpath(str(episode_idx))
            
            print(f"\n--- Episode {episode_idx} ---")
            print(f"Episode length: {episode_length}")
            print(f"Episode start: {episode_start}")
            print(f"Video directory: {episode_video_dir}")
            
            if not episode_video_dir.is_dir():
                print(f"Video directory not found for episode {episode_idx}")
                continue
            
            # Get video files for this episode
            episode_video_paths = sorted(episode_video_dir.glob('*.mp4'), key=lambda x: int(x.stem))
            print(f"Found {len(episode_video_paths)} video files: {[p.stem for p in episode_video_paths]}")
            
            # Process each camera
            for video_path in episode_video_paths:
                camera_idx = int(video_path.stem)
                camera_key = f'camera_{camera_idx}'
                
                if image_keys is not None and camera_key not in image_keys:
                    continue
                
                print(f"\n  Camera {camera_idx} ({camera_key}):")
                print(f"    Video file: {video_path.name}")
                
                # Get video resolution
                with av.open(str(video_path.absolute())) as container:
                    video = container.streams.video[0]
                    vcc = video.codec_context
                    resolution = (vcc.width, vcc.height)
                    fps = video.average_rate
                    duration = video.duration
                    print(f"    Resolution: {resolution}")
                    print(f"    FPS: {fps}")
                    print(f"    Duration: {duration}")
                
                # Read first few frames
                image_tf = get_image_transform(
                    input_res=resolution, 
                    output_res=resolution, 
                    bgr_to_rgb=False
                )
                
                frame_count = 0
                for step_idx, frame in enumerate(read_video(
                        video_path=str(video_path),
                        dt=dt,
                        img_transform=image_tf,
                        thread_type='FRAME',
                        thread_count=1
                    )):
                    if step_idx >= max_steps_per_episode:
                        break
                    
                    print(f"      Step {step_idx}: frame shape={frame.shape}, dtype={frame.dtype}")
                    print(f"        Frame range: [{frame.min()}, {frame.max()}]")
                    
                    # Save frame if requested
                    frame_count += 1
                
                print(f"    Read {frame_count} frames")


def main():
    if len(sys.argv) != 2:
        print("Usage: python read_DP_raw_data.py <dataset_path>")
        print("Example: python read_DP_raw_data.py /path/to/dataset")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    
    try:
        # Test reading with different configurations
        print("=" * 60)
        print("TEST 1: Read specific lowdim keys and camera images")
        print("=" * 60)
        read_raw_data(
            dataset_path, 
            lowdim_keys=['action', 'obs', 'timestamp'],
            image_keys=['camera_0', 'camera_1'],
            max_episodes=1, 
            max_steps_per_episode=3,
            save_frames=True,
            output_dir="saved_frames"
        )
        
    except Exception as e:
        print(f"Error reading data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
