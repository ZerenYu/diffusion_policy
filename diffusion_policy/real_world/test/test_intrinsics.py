#!/usr/bin/env python3
"""
Test script for intrinsics extraction from Orbbec frames
"""

import time
import numpy as np
from multiprocessing.managers import SharedMemoryManager
from ..single_orbbec import SingleOrbbec

def test_intrinsics():
    """Test intrinsics extraction"""
    
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
            capture_fps=30,
            enable_color=True,
            enable_depth=True,
            enable_infrared=False,
            verbose=True
        )
        
        try:
            # Start the camera
            print("Starting camera...")
            camera.start()
            
            # Wait for camera to be ready
            while not camera.is_ready:
                time.sleep(0.1)
            print("Camera is ready!")
            
            # Wait a bit for intrinsics to be extracted
            print("Waiting for intrinsics extraction...")
            time.sleep(2)
            
            # Test getting intrinsics
            print("Testing intrinsics...")
            intrinsics = camera.get_intrinsics()
            print(f"Intrinsics matrix:\n{intrinsics}")
            
            depth_scale = camera.get_depth_scale()
            print(f"Depth scale: {depth_scale}")
            
            # Test getting frames and verify data
            print("Testing frame capture...")
            for i in range(5):
                data = camera.get()
                if data is not None:
                    print(f"Frame {i}:")
                    if 'color' in data:
                        print(f"  Color shape: {data['color'].shape}")
                    if 'depth' in data:
                        print(f"  Depth shape: {data['depth'].shape}")
                    print(f"  Timestamp: {data['timestamp']}")
                else:
                    print(f"Frame {i}: No data received")
                
                time.sleep(0.5)
            
        finally:
            # Stop the camera
            print("Stopping camera...")
            camera.stop()
            print("Camera stopped")

if __name__ == "__main__":
    test_intrinsics() 