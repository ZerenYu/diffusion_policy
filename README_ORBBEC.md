# SingleOrbbec Implementation

This module provides a `SingleOrbbec` class that mimics the functionality of `SingleRealsense` but for Orbbec cameras. It supports color, depth, and infrared streams, recording capabilities, and real-time frame processing.

## Features

- **Multi-stream support**: Color, depth, and infrared streams
- **Real-time processing**: Shared memory ring buffers for efficient data transfer
- **Recording**: Video recording with H.264 encoding
- **Device management**: Automatic device detection and configuration
- **Software alignment**: Intelligent stream profile selection for optimal resolution matching
- **Thread safety**: Multi-process architecture with shared memory
- **Transform support**: Custom data transformations for processing pipelines

## Dependencies

- `pyorbbecsdk`: Orbbec SDK for Python
- `opencv-python`: Computer vision library
- `numpy`: Numerical computing
- `multiprocessing`: Multi-process support
- `threadpoolctl`: Thread pool control

## Usage

### Basic Usage

```python
from multiprocessing.managers import SharedMemoryManager
from diffusion_policy.real_world.single_orbbec import SingleOrbbec

# Get connected devices
devices = SingleOrbbec.get_connected_devices_serial()
print(f"Connected devices: {devices}")

# Create shared memory manager
with SharedMemoryManager() as shm_manager:
    # Create camera instance
            camera = SingleOrbbec(
            shm_manager=shm_manager,
            device_id=devices[0],
            resolution=(1280, 800),
            capture_fps=30,
        enable_color=True,
        enable_depth=True,
        enable_infrared=False,
        verbose=True
    )
    
    # Start camera
    camera.start()
    
    # Wait for camera to be ready
    while not camera.is_ready:
        time.sleep(0.1)
    
    # Get frames
    for i in range(10):
        data = camera.get()
        if data is not None:
            print(f"Frame {i}: Color shape {data['color'].shape}")
    
    # Stop camera
    camera.stop()
```

### Context Manager Usage

```python
with SharedMemoryManager() as shm_manager:
    with SingleOrbbec(
        shm_manager=shm_manager,
        device_id=devices[0],
        enable_color=True,
        enable_depth=True
    ) as camera:
        # Camera is automatically started and stopped
        data = camera.get()
        # Process data...
```

### Recording

```python
# Start recording
camera.start_recording("output.mp4")

# Stop recording
camera.stop_recording()
```

### Camera Settings

```python
# Set exposure (if supported by your device)
camera.set_exposure(exposure=1000, gain=1.0)

# Set white balance (if supported by your device)
camera.set_white_balance(white_balance=5000)
```

## API Reference

### SingleOrbbec Class

#### Constructor Parameters

- `shm_manager`: SharedMemoryManager instance
- `device_id`: Device serial number (optional)
- `resolution`: Camera resolution (width, height)
- `capture_fps`: Capture frame rate
- `put_fps`: Output frame rate (defaults to capture_fps)
- `put_downsample`: Whether to downsample output
- `record_fps`: Recording frame rate (defaults to capture_fps)
- `enable_color`: Enable color stream
- `enable_depth`: Enable depth stream
- `enable_infrared`: Enable infrared stream
- `get_max_k`: Maximum number of frames to buffer
- `transform`: Data transformation function
- `vis_transform`: Visualization transformation function
- `recording_transform`: Recording transformation function
- `video_recorder`: Custom video recorder
- `verbose`: Enable verbose logging

#### Methods

- `start(wait=True, put_start_time=None)`: Start the camera
- `stop(wait=True)`: Stop the camera
- `get(k=None, out=None)`: Get latest frame(s)
- `get_vis(out=None)`: Get visualization frame
- `set_exposure(exposure=None, gain=None)`: Set exposure settings
- `set_white_balance(white_balance=None)`: Set white balance
- `get_intrinsics()`: Get camera intrinsics matrix
- `get_depth_scale()`: Get depth scale factor
- `start_recording(video_path, start_time=-1)`: Start video recording
- `stop_recording()`: Stop video recording
- `restart_put(start_time)`: Restart frame output

#### Static Methods

- `get_connected_devices_serial()`: Get list of connected device serials

## Data Format

The camera returns frames in the following format:

```python
{
    'color': np.ndarray,           # BGR image (H, W, 3)
    'depth': np.ndarray,           # Depth image (H, W) - uint16
    'infrared': np.ndarray,        # IR image (H, W) - uint8
    'camera_capture_timestamp': float,  # Device timestamp
    'camera_receive_timestamp': float,  # Receive timestamp
    'timestamp': float,            # Processing timestamp
    'step_idx': int               # Frame index
}
```

## Examples

See the following example files:
- `real_world/test/single_orbbec_example.py`: Basic usage example
- `real_world/test/test_single_orbbec.py`: Test script
- `real_world/test/test_intrinsics.py`: Intrinsics extraction test

## Notes

1. **Device Compatibility**: This implementation is designed for Orbbec cameras that support the pyorbbecsdk. Different models may have different capabilities.

2. **IR Support**: Infrared support depends on your specific Orbbec device. Some devices support stereo IR, others mono IR.

3. **Exposure Control**: Exposure and gain control are implemented as placeholders. You may need to adapt these to your specific device's capabilities.

4. **Performance**: The implementation uses shared memory for efficient data transfer between processes. Adjust buffer sizes based on your performance requirements.

5. **Error Handling**: The implementation includes basic error handling, but you may need to add more robust error handling for your specific use case.

## Troubleshooting

1. **No devices found**: Ensure your Orbbec camera is properly connected and the SDK is installed.

2. **Frame drops**: Increase buffer sizes or reduce frame rates.

3. **Recording issues**: Ensure the output directory is writable and has sufficient disk space.

4. **Performance issues**: Adjust thread limits and buffer sizes based on your system capabilities. 