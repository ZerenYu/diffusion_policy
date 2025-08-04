from typing import Optional, Callable, Dict
import os
import enum
import time
import json
import numpy as np
import cv2
import multiprocessing as mp
from threadpoolctl import threadpool_limits
from multiprocessing.managers import SharedMemoryManager
from pyorbbecsdk import *
from diffusion_policy.common.timestamp_accumulator import get_accumulate_timestamp_idxs
from .orbbec_utils import frame_to_bgr_image, process_depth_frame, process_ir_frame
from diffusion_policy.shared_memory.shared_ndarray import SharedNDArray
from diffusion_policy.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from diffusion_policy.shared_memory.shared_memory_queue import SharedMemoryQueue, Full, Empty
from diffusion_policy.real_world.video_recorder import VideoRecorder

class Command(enum.Enum):
    SET_COLOR_OPTION = 0
    SET_DEPTH_OPTION = 1
    START_RECORDING = 2
    STOP_RECORDING = 3
    RESTART_PUT = 4

class SingleOrbbec(mp.Process):
    MAX_PATH_LENGTH = 4096 # linux path has a limit of 4096 bytes

    def __init__(
            self, 
            shm_manager: SharedMemoryManager,
            device_id: str = None,
            resolution=(640,480),
            capture_fps=30,
            put_fps=None,
            put_downsample=True,
            record_fps=None,
            enable_color=True,
            enable_depth=True,
            get_max_k=30,
            transform: Optional[Callable[[Dict], Dict]] = None,
            vis_transform: Optional[Callable[[Dict], Dict]] = None,
            recording_transform: Optional[Callable[[Dict], Dict]] = None,
            video_recorder: Optional[VideoRecorder] = None,
            verbose=False
        ):
        super().__init__()

        if put_fps is None:
            put_fps = capture_fps
        if record_fps is None:
            record_fps = capture_fps

        # create ring buffer
        resolution = tuple(resolution)
        shape = resolution[::-1]
        examples = dict()
        if enable_color:
            examples['color'] = np.empty(
                shape=shape+(3,), dtype=np.uint8)
        if enable_depth:
            examples['depth'] = np.empty(
                shape=shape, dtype=np.uint16)

        examples['camera_capture_timestamp'] = 0.0
        examples['camera_receive_timestamp'] = 0.0
        examples['timestamp'] = 0.0
        examples['step_idx'] = 0

        vis_ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=examples if vis_transform is None 
                else vis_transform(dict(examples)),
            get_max_k=1,
            get_time_budget=0.2,
            put_desired_frequency=capture_fps
        )

        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=examples if transform is None
                else transform(dict(examples)),
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=put_fps
        )

        # create command queue
        examples = {
            'cmd': Command.SET_COLOR_OPTION.value,
            'option_enum': 0,  # Placeholder for Orbbec options
            'option_value': 0.0,
            'video_path': np.array('a'*self.MAX_PATH_LENGTH),
            'recording_start_time': 0.0,
            'put_start_time': 0.0
        }

        command_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=examples,
            buffer_size=128
        )

        # create shared array for intrinsics
        intrinsics_array = SharedNDArray.create_from_shape(
                mem_mgr=shm_manager,
                shape=(7,),
                dtype=np.float64)
        intrinsics_array.get()[:] = 0

        # create video recorder
        if video_recorder is None:
            # Orbbec uses bgr24 pixel format
            # default thread_type to FRAME
            # i.e. each frame uses one core
            # instead of all cores working on all frames.
            # this prevents CPU over-subscription and
            # improves performance significantly
            video_recorder = VideoRecorder.create_h264(
                fps=record_fps, 
                codec='h264',
                input_pix_fmt='bgr24', 
                crf=18,
                thread_type='FRAME',
                thread_count=1)

        # copied variables
        self.device_id = device_id
        self.resolution = resolution
        self.capture_fps = capture_fps
        self.put_fps = put_fps
        self.put_downsample = put_downsample
        self.record_fps = record_fps
        self.enable_color = enable_color
        self.enable_depth = enable_depth

        self.transform = transform
        self.vis_transform = vis_transform
        self.recording_transform = recording_transform
        self.video_recorder = video_recorder
        self.verbose = verbose
        self.put_start_time = None

        # shared variables
        self.stop_event = mp.Event()
        self.ready_event = mp.Event()
        self.ring_buffer = ring_buffer
        self.vis_ring_buffer = vis_ring_buffer
        self.command_queue = command_queue
        self.intrinsics_array = intrinsics_array
    
    def get_sw_align_config(self, pipeline, color_req_width=None, color_req_height=None, depth_req_width=None, depth_req_height=None, fps=15):
        """
        Use software alignment to configure the flow.
        Priority should be given to using the command-line parameters color_req_width/color_req_height and depth_req_width/depth_req_height
        If not provided, use global COLOR_CAMERA_WIDTH/COLOR_CAMERA_HEIGHT and DEPTH_CAMERA_WIDTH/DEPTH_CAMERA_HEIGHT
        If the final resolution exists and matches the profile, use the specified resolution; otherwise, use the default profile and print the prompt.
        """
        # Determine final resolution: prioritize command line, then global
        cw = color_req_width if color_req_width is not None else self.resolution[0]
        ch = color_req_height if color_req_height is not None else self.resolution[1]
        dw = depth_req_width if depth_req_width is not None else self.resolution[0]
        dh = depth_req_height if depth_req_height is not None else self.resolution[1]

        config = Config()

        
        try:
            color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)

            # Select color profile
            color_profile = None
            if cw and ch and self.enable_color:
                for cp in color_profiles:
                    if cp.get_format() == OBFormat.RGB and cp.get_width() == cw and cp.get_height() == ch and cp.get_fps() == fps:
                        color_profile = cp
                        print(f"[zyu] color_profile is {color_profile}")
                        if self.verbose:
                            print(f"Use specified color resolution profile: {cw}x{ch} @ {fps}fps")
                        break
                config.enable_stream(color_profile)

            # Select depth profile
            depth_profile = None
            if dw and dh and self.enable_depth:
                for dp in depth_profiles:
                    if dp.get_width() == dw and dp.get_height() == dh and dp.get_fps() == fps:
                        depth_profile = dp
                        print(f"[zyu] depth_profile is {depth_profile}")
                        if self.verbose:
                            print(f"Use specified depth resolution profile: {dw}x{dh} @ {fps}fps")
                        break
                config.enable_stream(depth_profile)

        except Exception as e:
            if self.verbose:
                print(f"Failed to get software align config: {e}")
            return None

        return config
    
    @staticmethod
    def get_connected_devices_serial():
        """Get list of connected Orbbec devices"""
        serials = list()
        try:
            # Get device list from Orbbec SDK
            context = Context()
            device_list = context.query_devices()
            for device in device_list:
                device_info = device.get_device_info()
                serial = device_info.get_serial_number()
                if serial:
                    print(f"[zyu] serial is {serial}")
                    serials.append(serial)
                    print(f"[zyu] serial is {serial} append fail")
        except Exception as e:
            print(f"Error getting Orbbec devices: {e}")
        serials = sorted(serials)
        return serials

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= user API ===========
    def start(self, wait=True, put_start_time=None):
        self.put_start_time = put_start_time
        super().start()
        if wait:
            self.start_wait()
    
    def stop(self, wait=True):
        self.stop_event.set()
        if wait:
            self.end_wait()

    def start_wait(self):
        self.ready_event.wait()
    
    def end_wait(self):
        self.join()

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    def get(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k, out=out)
    
    def get_vis(self, out=None):
        return self.vis_ring_buffer.get(out=out)
    
    # ========= user API ===========
    def set_color_option(self, option: int, value: float):
        """Set color sensor option (placeholder for Orbbec-specific options)"""
        self.command_queue.put({
            'cmd': Command.SET_COLOR_OPTION.value,
            'option_enum': option,
            'option_value': value
        })
    
    def set_exposure(self, exposure=None, gain=None):
        """
        exposure: exposure time in microseconds
        gain: gain value
        Note: Orbbec exposure/gain control may differ from RealSense
        """
        if exposure is None and gain is None:
            # auto exposure - implement based on Orbbec SDK
            pass
        else:
            # manual exposure - implement based on Orbbec SDK
            if exposure is not None:
                self.set_color_option(0, exposure)  # Placeholder option
            if gain is not None:
                self.set_color_option(1, gain)  # Placeholder option
    
    def set_white_balance(self, white_balance=None):
        """Set white balance (placeholder for Orbbec-specific implementation)"""
        if white_balance is None:
            self.set_color_option(2, 1.0)  # Auto white balance
        else:
            self.set_color_option(2, 0.0)  # Manual white balance
            self.set_color_option(3, white_balance)

    def get_intrinsics(self):
        """Get camera intrinsics matrix"""
        assert self.ready_event.is_set()
        fx, fy, ppx, ppy = self.intrinsics_array.get()[:4]
        mat = np.eye(3)
        mat[0,0] = fx
        mat[1,1] = fy
        mat[0,2] = ppx
        mat[1,2] = ppy
        return mat

    def get_depth_scale(self):
        """Get depth scale factor"""
        assert self.ready_event.is_set()
        scale = self.intrinsics_array.get()[-1]
        return scale
    
    def start_recording(self, video_path: str, start_time: float=-1):
        assert self.enable_color

        path_len = len(video_path.encode('utf-8'))
        if path_len > self.MAX_PATH_LENGTH:
            raise RuntimeError('video_path too long.')
        self.command_queue.put({
            'cmd': Command.START_RECORDING.value,
            'video_path': video_path,
            'recording_start_time': start_time
        })
        
    def stop_recording(self):
        self.command_queue.put({
            'cmd': Command.STOP_RECORDING.value
        })
    
    def restart_put(self, start_time):
        self.command_queue.put({
            'cmd': Command.RESTART_PUT.value,
            'put_start_time': start_time
        })
     
    # ========= internal API ===========
    def run(self):
        # limit threads
        threadpool_limits(1)
        cv2.setNumThreads(1)

        w, h = self.resolution
        fps = self.capture_fps
        
        # Setup Orbbec pipeline
        pipeline = Pipeline()
        device = pipeline.get_device()

        # Use software alignment configuration
        config = self.get_sw_align_config(pipeline)
        

        try:
            pipeline.start(config)
            align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
            # Initialize intrinsics extraction flag
            intrinsics_extracted = False
            
            if self.verbose:
                print(f'[SingleOrbbec {self.device_id}] Main loop started.')

            # put frequency regulation
            put_idx = None
            put_start_time = self.put_start_time
            if put_start_time is None:
                put_start_time = time.time()

            iter_idx = 0
            t_start = time.time()
            while not self.stop_event.is_set():
                # wait for frames to come in
                frames = pipeline.wait_for_frames(100)
                if not frames:
                    print(f"[zyu] no frames")
                    continue
                # frames = align_filter.process(frames)
                receive_time = time.time()

                # grab data
                data = dict()
                data['camera_receive_timestamp'] = receive_time
                data['camera_capture_timestamp'] = receive_time  # Orbbec doesn't provide device timestamp
                
                # Extract intrinsics from first frames if not done yet
                if not intrinsics_extracted:
                    if self.enable_color:
                        color_frame = frames.get_color_frame()
                        print(f"[zyu] color_frame is {color_frame is not None}")
                        if color_frame:
                            color_data = frame_to_bgr_image(color_frame)
                        else:
                            print(f"[zyu] no color frame")
                            continue
                    if self.enable_depth:
                        depth_frame = frames.get_depth_frame()
                        print(f"[zyu] depth_frame is {depth_frame is not None}")
                        if depth_frame:
                            depth_frame = depth_frame.as_video_frame()
                            depth_profile = depth_frame.get_stream_profile()
                            depth_intrinsics = depth_profile.as_video_stream_profile().get_intrinsic()
                            # Store depth scale in the last element (placeholder for now)
                            # You might need to get this from the device or sensor
                            self.intrinsics_array.get()[-1] = 0.001  # Default depth scale
                        else:
                            print(f"[zyu] no depth frame")
                            continue
                    intrinsics_extracted = True
                    if self.verbose:
                        print(f'[SingleOrbbec {self.device_id}] Intrinsics extracted.')
                
                if self.enable_color:
                    color_frame = frames.get_color_frame()
                    if color_frame:
                        color_data = frame_to_bgr_image(color_frame)
                        if color_data is not None:
                            data['color'] = color_data
                
                if self.enable_depth:
                    depth_frame = frames.get_depth_frame()
                    if depth_frame:
                        depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
                        depth_data = depth_data.reshape(depth_frame.get_height(), depth_frame.get_width())
                        data['depth'] = depth_data
                
                
                # apply transform
                put_data = data
                if self.transform is not None:
                    put_data = self.transform(dict(data))

                if self.put_downsample:                
                    # put frequency regulation
                    print(f"[zyu] inputs are {receive_time}, {put_start_time}, {self.put_fps}, {put_idx}")
                    local_idxs, global_idxs, put_idx \
                        = get_accumulate_timestamp_idxs(
                            timestamps=[receive_time],
                            start_time=put_start_time,
                            dt=1/self.put_fps,
                            next_global_idx=put_idx,
                            allow_negative=True
                        )
                    print(f"[zyu] global_idxs is {global_idxs}")
                    for step_idx in global_idxs:
                        try:
                            put_data['step_idx'] = step_idx
                            put_data['timestamp'] = receive_time
                            self.ring_buffer.put(put_data, wait=False)
                        except TimeoutError as e:
                            if self.verbose:
                                print(f"[SingleOrbbec {self.device_id}] Ring buffer full, skipping frame")
                            continue
                else:
                    step_idx = int((receive_time - put_start_time) * self.put_fps)
                    put_data['step_idx'] = step_idx
                    put_data['timestamp'] = receive_time
                    self.ring_buffer.put(put_data, wait=False)

                # signal ready
                if iter_idx == 0:
                    self.ready_event.set()
                
                # put to vis
                vis_data = data
                if self.vis_transform == self.transform:
                    vis_data = put_data
                elif self.vis_transform is not None:
                    vis_data = self.vis_transform(dict(data))
                try:
                    self.vis_ring_buffer.put(vis_data, wait=False)
                except TimeoutError as e:
                    if self.verbose:
                        print(f"[SingleOrbbec {self.device_id}] Vis ring buffer full, skipping frame")
                    continue
                
                # record frame
                rec_data = data
                if self.recording_transform == self.transform:
                    rec_data = put_data
                elif self.recording_transform is not None:
                    rec_data = self.recording_transform(dict(data))

                if self.video_recorder.is_ready() and 'color' in rec_data:
                    self.video_recorder.write_frame(rec_data['color'], 
                        frame_time=receive_time)

                # perf
                t_end = time.time()
                duration = t_end - t_start
                frequency = np.round(1 / duration, 1)
                t_start = t_end
                if self.verbose:
                    print(f'[SingleOrbbec {self.device_id}] FPS {frequency}')

                # fetch command from queue
                try:
                    commands = self.command_queue.get_all()
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0

                # execute commands
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']
                    if cmd == Command.SET_COLOR_OPTION.value:
                        # Implement Orbbec-specific color options
                        option = command['option_enum']
                        value = float(command['option_value'])
                        # Placeholder for Orbbec sensor option setting
                        pass
                    elif cmd == Command.SET_DEPTH_OPTION.value:
                        # Implement Orbbec-specific depth options
                        option = command['option_enum']
                        value = float(command['option_value'])
                        # Placeholder for Orbbec sensor option setting
                        pass
                    elif cmd == Command.START_RECORDING.value:
                        video_path = str(command['video_path'])
                        start_time = command['recording_start_time']
                        if start_time < 0:
                            start_time = None
                            print(f"[zyu] start_recording time is {start_time}")
                        self.video_recorder.start(video_path, start_time=start_time)
                    elif cmd == Command.STOP_RECORDING.value:
                        self.video_recorder.stop()
                        # soft-reset put to drop frames to prevent ring buffer overflow.
                        put_idx = None
                    elif cmd == Command.RESTART_PUT.value:
                        put_idx = None
                        put_start_time = command['put_start_time']

                iter_idx += 1
        finally:
            self.video_recorder.stop()
            pipeline.stop()
            self.ready_event.set()
        
        if self.verbose:
            print(f'[SingleOrbbec {self.device_id}] Exiting worker process.')
