#!/usr/bin/env python3
"""
Example usage of SingleOrbbec class
"""

import time
import cv2
import numpy as np
from multiprocessing.managers import SharedMemoryManager
from ..single_orbbec import SingleOrbbec

def main():
    """Example usage of SingleOrbbec"""
    
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
            enable_infrared=False,  # Set to True if your device supports IR
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
            
            # Create display window
            cv2.namedWindow("Orbbec Camera", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Orbbec Camera", 1280, 800)
            
            print("Press 'q' to quit, 'r' to start/stop recording")
            recording = False
            
            while True:
                # Get latest frame
                data = camera.get()
                if data is not None:
                    # Create display image
                    display = None
                    
                    if 'color' in data:
                        color_img = data['color']
                        if 'depth' in data:
                            # Normalize depth for visualization
                            depth_img = data['depth']
                            depth_normalized = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                            depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
                            
                            # Combine color and depth side by side
                            display = np.hstack((color_img, depth_colored))
                        else:
                            display = color_img
                    
                    if display is not None:
                        # Add recording indicator
                        if recording:
                            cv2.putText(display, "REC", (10, 30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        
                        cv2.imshow("Orbbec Camera", display)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    if not recording:
                        camera.start_recording(f"orbbec_recording_{int(time.time())}.mp4")
                        recording = True
                        print("Started recording")
                    else:
                        camera.stop_recording()
                        recording = False
                        print("Stopped recording")
            
        finally:
            # Stop recording if active
            if recording:
                camera.stop_recording()
            
            # Stop the camera
            print("Stopping camera...")
            camera.stop()
            cv2.destroyAllWindows()
            print("Camera stopped")

if __name__ == "__main__":
    main() 