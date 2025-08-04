#!/usr/bin/env python3
"""
Example usage of the joystick interface for teleopt control.
This script demonstrates how to connect to a joystick and read its state
for basic control applications.
"""

import time
import sys
import os

# Add the parent directory to the path to import joystick_control
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from joystick_control import JoystickInterface, JoystickButton, JoystickAxis


def example_basic_control():
    """Example of basic joystick control with continuous reading"""
    print("=== Basic Joystick Control Example ===")
    
    # Initialize joystick interface
    joystick = JoystickInterface()
    
    try:
        # Connect to the first available joystick
        if not joystick.connect_device(0):
            print("No joystick found. Please connect a joystick and try again.")
            return
        
        print("Joystick connected! Press Ctrl+C to exit.")
        print("Move the joystick or press buttons to see the output.")
        
        # Continuous reading loop
        while True:
            # Read current state
            state = joystick.read_state(0)
            
            if state:
                # Print axis values (only if they're not zero)
                active_axes = {k: v for k, v in state.axes.items() if abs(v) > 0.1}
                if active_axes:
                    print(f"Axes: {active_axes}")
                
                # Print button states (only pressed buttons)
                pressed_buttons = {k: v for k, v in state.buttons.items() if v}
                if pressed_buttons:
                    print(f"Buttons pressed: {pressed_buttons}")
                
                # Print hat states (only if not centered)
                active_hats = {k: v for k, v in state.hats.items() if v != (0, 0)}
                if active_hats:
                    print(f"Hats: {active_hats}")
            
            time.sleep(0.1)  # 10Hz reading rate
            
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        joystick.cleanup()


def example_robot_control():
    """Example of joystick control for robot movement"""
    print("=== Robot Control Example ===")
    
    joystick = JoystickInterface()
    
    try:
        if not joystick.connect_device(0):
            print("No joystick found. Please connect a joystick and try again.")
            return
        
        print("Robot control mode active!")
        print("Left stick: Move robot")
        print("Right stick: Rotate robot")
        print("A button: Emergency stop")
        print("B button: Reset position")
        print("Press Ctrl+C to exit.")
        
        while True:
            state = joystick.read_state(0)
            
            if state:
                # Get left stick values for movement
                left_x = state.axes.get(0, 0.0)  # Left stick X
                left_y = state.axes.get(1, 0.0)  # Left stick Y
                
                # Get right stick values for rotation
                right_x = state.axes.get(2, 0.0)  # Right stick X
                right_y = state.axes.get(3, 0.0)  # Right stick Y
                
                # Get trigger values
                left_trigger = state.axes.get(4, 0.0)  # Left trigger
                right_trigger = state.axes.get(5, 0.0)  # Right trigger
                
                # Check buttons
                a_button = state.buttons.get(0, False)  # A button
                b_button = state.buttons.get(1, False)  # B button
                
                # Simulate robot control commands
                if abs(left_x) > 0.1 or abs(left_y) > 0.1:
                    print(f"Move robot: X={left_x:.2f}, Y={left_y:.2f}")
                
                if abs(right_x) > 0.1 or abs(right_y) > 0.1:
                    print(f"Rotate robot: X={right_x:.2f}, Y={right_y:.2f}")
                
                if abs(left_trigger) > 0.1:
                    print(f"Left trigger: {left_trigger:.2f}")
                
                if abs(right_trigger) > 0.1:
                    print(f"Right trigger: {right_trigger:.2f}")
                
                if a_button:
                    print("EMERGENCY STOP!")
                
                if b_button:
                    print("Reset robot position")
            
            time.sleep(0.05)  # 20Hz for responsive control
            
    except KeyboardInterrupt:
        print("\nExiting robot control...")
    finally:
        joystick.cleanup()


def example_device_discovery():
    """Example of discovering and connecting to multiple joysticks"""
    print("=== Device Discovery Example ===")
    
    joystick = JoystickInterface()
    
    try:
        # Get number of available joysticks
        num_joysticks = len(joystick.get_connected_devices())
        print(f"Found {num_joysticks} joystick(s)")
        
        # Try to connect to all available joysticks
        for device_id in range(3):  # Try first 3 devices
            if joystick.connect_device(device_id):
                info = joystick.get_device_info(device_id)
                if info:
                    print(f"Connected to device {device_id}: {info['name']}")
                    print(f"  Axes: {info['num_axes']}, Buttons: {info['num_buttons']}, Hats: {info['num_hats']}")
        
        # Read from all connected devices
        connected_devices = joystick.get_connected_devices()
        if connected_devices:
            print(f"\nReading from {len(connected_devices)} connected device(s) for 3 seconds...")
            start_time = time.time()
            
            while time.time() - start_time < 3:
                for device_id in connected_devices:
                    state = joystick.read_state(device_id)
                    if state:
                        # Print only if there's significant input
                        active_axes = {k: v for k, v in state.axes.items() if abs(v) > 0.1}
                        pressed_buttons = {k: v for k, v in state.buttons.items() if v}
                        
                        if active_axes or pressed_buttons:
                            print(f"Device {device_id}: Axes={active_axes}, Buttons={pressed_buttons}")
                
                time.sleep(0.1)
        else:
            print("No joysticks connected")
            
    except KeyboardInterrupt:
        print("\nExiting device discovery...")
    finally:
        joystick.cleanup()


def main():
    """Main function to run examples"""
    print("Joystick Interface Examples")
    print("==========================")
    print("1. Basic control example")
    print("2. Robot control example")
    print("3. Device discovery example")
    print("4. Run all examples")
    
    try:
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            example_basic_control()
        elif choice == "2":
            example_robot_control()
        elif choice == "3":
            example_device_discovery()
        elif choice == "4":
            example_basic_control()
            print("\n" + "="*50 + "\n")
            example_robot_control()
            print("\n" + "="*50 + "\n")
            example_device_discovery()
        else:
            print("Invalid choice. Running basic control example...")
            example_basic_control()
            
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main() 