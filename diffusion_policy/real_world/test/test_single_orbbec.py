#!/usr/bin/env python3
"""
Test script for SingleOrbbec implementation
"""

import time
import os
import cv2
import numpy as np
from multiprocessing.managers import SharedMemoryManager
from diffusion_policy.real_world.single_orbbec import SingleOrbbec

def test_single_orbbec():
    """Test the SingleOrbbec implementation"""
    
    # Create test folder for saving images
    test_folder = "test_images"
    os.makedirs(test_folder, exist_ok=True)
    print(f"Created test folder: {test_folder}")
    
    # Get connected devices
    devices = SingleOrbbec.get_connected_devices_serial()
    print(f"Connected Orbbec devices: {devices}")
    
    if not devices:
        print("No Orbbec devices found. Please connect a device and try again.")
        return
    
    # Use the first available device
    device_id = devices[0]
    print(f"Using device: {device_id}")
    
    # Create shared memory manager
    with SharedMemoryManager() as shm_manager:
        # Create SingleOrbbec instance
        camera = SingleOrbbec(
            shm_manager=shm_manager,
            device_id=device_id,
            resolution=(1280, 800),
            capture_fps=15,
            enable_color=True,
            enable_depth=True,
            verbose=True
        )
        
        try:
            # Start the camera
            print("Starting camera...")
            camera.start()
            iter_count = 0
            # Wait for camera to be ready
            while iter_count < 10:
                time.sleep(0.1)
                iter_count += 1
            print("Camera is ready!")
            
            # Test getting frames
            print("Testing frame capture...")
            tenth_frame_data = None
            for i in range(10):
                # Get latest frame
                data = camera.get()
                if data is not None:
                    print(f"Frame {i}:")
                    if 'color' in data:
                        print(f"  Color shape: {data['color'].shape}")
                    if 'depth' in data:
                        print(f"  Depth shape: {data['depth'].shape}")
                    if 'infrared' in data:
                        print(f"  Infrared shape: {data['infrared'].shape}")
                    print(f"  Timestamp: {data['timestamp']}")
                    print(f"  Step idx: {data['step_idx']}")
                    
                    # Save the tenth frame (index 9)
                    if i == 10:
                        tenth_frame_data = data
                        print(f"  *** Tenth frame captured for saving ***")
                else:
                    print(f"Frame {i}: No data received")
                
                time.sleep(0.1)
            
            # Save the tenth frame images
            if tenth_frame_data is not None:
                print("Saving tenth frame images...")
                
                if 'color' in tenth_frame_data:
                    color_path = os.path.join(test_folder, "tenth_frame_color.jpg")
                    cv2.imwrite(color_path, cv2.cvtColor(tenth_frame_data['color'], cv2.COLOR_RGB2BGR))
                    print(f"Saved color image to: {color_path}")
                
                if 'depth' in tenth_frame_data:
                    depth_path = os.path.join(test_folder, "tenth_frame_depth.png")
                    # Normalize depth for visualization (0-255)
                    depth_normalized = ((tenth_frame_data['depth'] - tenth_frame_data['depth'].min()) / 
                                      (tenth_frame_data['depth'].max() - tenth_frame_data['depth'].min()) * 255).astype(np.uint8)
                    cv2.imwrite(depth_path, depth_normalized)
                    print(f"Saved depth image to: {depth_path}")
                
                if 'infrared' in tenth_frame_data:
                    infrared_path = os.path.join(test_folder, "tenth_frame_infrared.png")
                    cv2.imwrite(infrared_path, tenth_frame_data['infrared'])
                    print(f"Saved infrared image to: {infrared_path}")
                
                print("Tenth frame images saved successfully!")
            else:
                print("No tenth frame data available for saving")
            
            # Test intrinsics
            print("Testing intrinsics...")
            intrinsics = camera.get_intrinsics()
            print(f"Intrinsics matrix:\n{intrinsics}")
            
            depth_scale = camera.get_depth_scale()
            print(f"Depth scale: {depth_scale}")
            
            # Test recording
            print("Testing recording...")
            camera.start_recording("test_recording.mp4")
            time.sleep(60)
            camera.stop_recording()
            print("Recording test completed")
            
        finally:
            # Stop the camera
            print("Stopping camera...")
            camera.stop()
            print("Camera stopped")

if __name__ == "__main__":
    test_single_orbbec() 