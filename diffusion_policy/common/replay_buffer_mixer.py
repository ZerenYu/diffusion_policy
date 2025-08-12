from typing import Union, Dict, Optional, List
import os
import shutil
import pathlib
import zarr
import numpy as np

from replay_buffer import ReplayBuffer

class ReplayBufferMixer(ReplayBuffer):
    """
    A ReplayBuffer that is created by mixing episodes from other ReplayBuffers.
    """

    def __init__(self,
            src_buffers: List[ReplayBuffer],
            episode_indices: Optional[List[List[int]]] = None,
            dest_buffer: Optional[ReplayBuffer] = None):
        """
        Mixes episodes from a list of source ReplayBuffers into a destination ReplayBuffer.
        If dest_buffer is not provided, a new in-memory buffer is created.

        The ReplayBufferMixer instance itself becomes the mixed replay buffer.

        :param src_buffers: A list of ReplayBuffer instances to source episodes from.
        :param episode_indices: A list of lists of integers. The outer list corresponds to the
                                src_buffers, and the inner list contains the episode indices
                                to be taken from that buffer. If None, all episodes from all
                                source buffers are used.
        :param dest_buffer: The ReplayBuffer instance to which the mixed episodes will be added.
                            If None, a new in-memory (numpy-backed) ReplayBuffer is created.
        """
        if dest_buffer is None:
            dest_buffer = ReplayBuffer.create_empty_numpy()

        if episode_indices is None:
            episode_indices = [list(range(buf.n_episodes)) for buf in src_buffers]

        assert len(src_buffers) == len(episode_indices)

        for src_buffer, indices in zip(src_buffers, episode_indices):
            for episode_idx in indices:
                episode = src_buffer.get_episode(episode_idx, copy=True)
                dest_buffer.add_episode(episode)

        super().__init__(root=dest_buffer.root)

    @classmethod
    def create_from_directories(cls,
            src_dirs: List[str],
            dest_dir: str,
            episode_indices: List[List[int]],
            src_buffer_kwargs: Optional[dict] = None):
        """
        Creates a new dataset by selecting episodes from source directories.
        Each source directory must contain 'replay_buffer.zarr' and a 'videos' directory.
        Copies both the replay buffer data and corresponding videos.

        :param src_dirs: List of paths to the source directories.
        :param dest_dir: Path to save the new dataset. Overwritten if it exists.
        :param episode_indices: List of lists of episode indices to select.
        :param src_buffer_kwargs: Kwargs for opening the source buffers.
        """
        dest_dir = pathlib.Path(os.path.expanduser(dest_dir))
        dest_zarr_path = dest_dir.joinpath('replay_buffer.zarr')
        dest_videos_path = dest_dir.joinpath('videos')

        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        dest_dir.mkdir(parents=True)
        dest_videos_path.mkdir(exist_ok=True)

        if src_buffer_kwargs is None:
            src_buffer_kwargs = {'mode': 'r'}
        
        src_buffers = []
        for src_dir in src_dirs:
            src_zarr_path = pathlib.Path(os.path.expanduser(src_dir)).joinpath('replay_buffer.zarr')
            if not src_zarr_path.exists():
                raise FileNotFoundError(f"Source replay buffer not found at {src_zarr_path}")
            src_buffers.append(ReplayBuffer.create_from_path(str(src_zarr_path), **src_buffer_kwargs))

        dest_buffer = ReplayBuffer.create_from_path(str(dest_zarr_path), mode='w')

        mixer = cls(
            src_buffers=src_buffers,
            episode_indices=episode_indices,
            dest_buffer=dest_buffer
        )

        # Copy videos, mapping old episode indices to new ones
        new_episode_idx = 0
        for src_idx, indices_from_this_src in enumerate(episode_indices):
            src_videos_path = pathlib.Path(os.path.expanduser(src_dirs[src_idx])).joinpath('videos')
            if not src_videos_path.is_dir():
                print(f"Warning: Source videos directory not found at {src_videos_path}. Videos for this source will not be copied.")
                new_episode_idx += len(indices_from_this_src)
                continue

            for old_episode_idx in indices_from_this_src:
                src_video_dir = src_videos_path.joinpath(str(old_episode_idx))
                dest_video_dir = dest_videos_path.joinpath(str(new_episode_idx))
                if src_video_dir.is_dir():
                    shutil.copytree(src_video_dir, dest_video_dir)
                else:
                    print(f"Warning: Video directory for episode {old_episode_idx} in {src_dirs[src_idx]} not found, skipping.")
                new_episode_idx += 1
        
        return mixer


if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Create a new dataset by selecting episodes and copying corresponding videos from source dataset directories.")

    parser.add_argument('--src-dirs', nargs='+', required=True,
                        help="Paths to the source dataset directories. Each must contain replay_buffer.zarr and optionally a videos/ directory.")
    parser.add_argument('--dest-dir', type=str, required=True,
                        help="Path to save the new dataset directory.")
    parser.add_argument('--indices', nargs='+', required=True,
                        help="Episode indices to select. Format: 'SRC_IDX,START:END'. "
                             "Can be specified multiple times. SRC_IDX is 0-based. "
                             "Example: --indices 0,0:10 0,20:30 1,11:21")

    args = parser.parse_args()

    episode_indices = [[] for _ in args.src_dirs]
    try:
        for spec in args.indices:
            src_idx_str, range_str = spec.split(',', 1)
            src_idx = int(src_idx_str)  # User provides 0-based index

            if not (0 <= src_idx < len(args.src_dirs)):
                raise ValueError(f"Source index {src_idx} is out of bounds for {len(args.src_dirs)} source directories.")

            start_str, end_str = range_str.split(':', 1)
            start = int(start_str)
            end = int(end_str)
            if start > end:
                raise ValueError(f"Start index cannot be greater than end index: {start}:{end}")
            episode_indices[src_idx].extend(range(start, end + 1))
        
        print(f"[zyu] episode_indices: {episode_indices}")
    except ValueError as e:
        print(f"Error parsing --indices: {e}", file=sys.stderr)
        print("Expected format: 'SRC_IDX,START:END', e.g., '0,0:10 0,20:30'", file=sys.stderr)
        sys.exit(1)

    print(f"Source directories: {args.src_dirs}")
    print(f"Destination directory: {args.dest_dir}")
    print(f"Episode indices to copy: {episode_indices}")

    ReplayBufferMixer.create_from_directories(
        src_dirs=args.src_dirs,
        dest_dir=args.dest_dir,
        episode_indices=episode_indices
    )

    print("Done.")
