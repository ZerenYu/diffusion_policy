import pygame
import time
import threading
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class JoystickButton(Enum):
    """Standard joystick button mappings"""
    A = 0
    B = 1
    X = 2
    Y = 3
    LEFT_BUMPER = 4
    RIGHT_BUMPER = 5
    BACK = 6
    START = 7
    LEFT_STICK = 8
    RIGHT_STICK = 9
    MANU = 11


class JoystickAxis(Enum):
    """Standard joystick axis mappings"""
    LEFT_X = 0
    LEFT_Y = 1
    RIGHT_X = 2
    RIGHT_Y = 3
    LEFT_TRIGGER = 4
    RIGHT_TRIGGER = 5


@dataclass
class JoystickState:
    """Represents the current state of a joystick"""
    axes: Dict[int, float]  # axis_id -> value (-1.0 to 1.0)
    buttons: Dict[int, bool]  # button_id -> pressed state
    hats: Dict[int, Tuple[int, int]]  # hat_id -> (x, y) position
    timestamp: float
    connected: bool


class JoystickInterface:
    """
    Basic joystick interface using pygame for device connection and state reading.
    Provides functions to connect to joystick devices and read their current state.
    """
    
    def __init__(self):
        """Initialize the joystick interface"""
        self.joysticks: Dict[int, pygame.joystick.Joystick] = {}
        self.states: Dict[int, JoystickState] = {}
        self.connected_devices: List[int] = []
        self._running = False
        self._lock = threading.Lock()
        
        # Initialize pygame
        pygame.init()
        pygame.joystick.init()
        
        print(f"Joystick interface initialized. Found {pygame.joystick.get_count()} joystick(s)")
    
    def connect_device(self, device_id: int = 0) -> bool:
        """
        Connect to a joystick device by ID.
        
        Args:
            device_id: The ID of the joystick device (0-based)
            
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            if device_id >= pygame.joystick.get_count():
                print(f"Error: Joystick device {device_id} not found. Available devices: 0-{pygame.joystick.get_count()-1}")
                return False
            
            # Create joystick object
            joystick = pygame.joystick.Joystick(device_id)
            joystick.init()
            
            # Get joystick info
            name = joystick.get_name()
            num_axes = joystick.get_numaxes()
            num_buttons = joystick.get_numbuttons()
            num_hats = joystick.get_numhats()
            
            print(f"Connected to joystick {device_id}: {name}")
            print(f"  Axes: {num_axes}, Buttons: {num_buttons}, Hats: {num_hats}")
            
            # Store joystick and initialize state
            with self._lock:
                self.joysticks[device_id] = joystick
                self.connected_devices.append(device_id)
                
                # Initialize state
                axes = {i: 0.0 for i in range(num_axes)}
                buttons = {i: False for i in range(num_buttons)}
                hats = {i: (0, 0) for i in range(num_hats)}
                
                self.states[device_id] = JoystickState(
                    axes=axes,
                    buttons=buttons,
                    hats=hats,
                    timestamp=time.time(),
                    connected=True
                )
            
            return True
            
        except Exception as e:
            print(f"Error connecting to joystick {device_id}: {e}")
            return False
    
    def disconnect_device(self, device_id: int) -> bool:
        """
        Disconnect a joystick device.
        
        Args:
            device_id: The ID of the joystick device to disconnect
            
        Returns:
            bool: True if disconnection successful, False otherwise
        """
        try:
            with self._lock:
                if device_id in self.joysticks:
                    joystick = self.joysticks[device_id]
                    joystick.quit()
                    del self.joysticks[device_id]
                    
                    if device_id in self.states:
                        self.states[device_id].connected = False
                    
                    if device_id in self.connected_devices:
                        self.connected_devices.remove(device_id)
                    
                    print(f"Disconnected joystick {device_id}")
                    return True
                else:
                    print(f"Joystick {device_id} not found")
                    return False
                    
        except Exception as e:
            print(f"Error disconnecting joystick {device_id}: {e}")
            return False
    
    def read_state(self, device_id: int = 0) -> Optional[JoystickState]:
        """
        Read the current state of a joystick device.
        
        Args:
            device_id: The ID of the joystick device to read
            
        Returns:
            Optional[JoystickState]: Current joystick state or None if device not found
        """
        try:
            with self._lock:
                if device_id not in self.joysticks or device_id not in self.states:
                    return None
                
                joystick = self.joysticks[device_id]
                state = self.states[device_id]
                
                # Process pygame events to update joystick state
                pygame.event.pump()
                
                # Update axes
                for axis_id in range(joystick.get_numaxes()):
                    value = joystick.get_axis(axis_id)
                    # Apply dead zone to prevent drift
                    if abs(value) < 0.1:
                        value = 0.0
                    state.axes[axis_id] = value
                
                # Update buttons
                for button_id in range(joystick.get_numbuttons()):
                    state.buttons[button_id] = joystick.get_button(button_id)
                
                # Update hats
                for hat_id in range(joystick.get_numhats()):
                    state.hats[hat_id] = joystick.get_hat(hat_id)
                
                # Update timestamp
                state.timestamp = time.time()
                
                return state
                
        except Exception as e:
            print(f"Error reading joystick {device_id} state: {e}")
            return None
    
    def get_connected_devices(self) -> List[int]:
        """
        Get list of connected joystick device IDs.
        
        Returns:
            List[int]: List of connected device IDs
        """
        with self._lock:
            return self.connected_devices.copy()
    
    def get_device_info(self, device_id: int) -> Optional[Dict]:
        """
        Get information about a joystick device.
        
        Args:
            device_id: The ID of the joystick device
            
        Returns:
            Optional[Dict]: Device information or None if device not found
        """
        try:
            with self._lock:
                if device_id not in self.joysticks:
                    return None
                
                joystick = self.joysticks[device_id]
                
                return {
                    'name': joystick.get_name(),
                    'num_axes': joystick.get_numaxes(),
                    'num_buttons': joystick.get_numbuttons(),
                    'num_hats': joystick.get_numhats(),
                    'connected': device_id in self.connected_devices
                }
                
        except Exception as e:
            print(f"Error getting device info for joystick {device_id}: {e}")
            return None
    
    def get_axis_value(self, device_id: int, axis_id: int) -> float:
        """
        Get the value of a specific axis.
        
        Args:
            device_id: The ID of the joystick device
            axis_id: The ID of the axis
            
        Returns:
            float: Axis value (-1.0 to 1.0) or 0.0 if not found
        """
        state = self.read_state(device_id)
        if state and axis_id in state.axes:
            return state.axes[axis_id]
        return 0.0
    
    def get_button_state(self, device_id: int, button_id: int) -> bool:
        """
        Get the state of a specific button.
        
        Args:
            device_id: The ID of the joystick device
            button_id: The ID of the button
            
        Returns:
            bool: Button pressed state or False if not found
        """
        state = self.read_state(device_id)
        if state and button_id in state.buttons:
            return state.buttons[button_id]
        return False
    
    def get_hat_state(self, device_id: int, hat_id: int) -> Tuple[int, int]:
        """
        Get the state of a specific hat.
        
        Args:
            device_id: The ID of the joystick device
            hat_id: The ID of the hat
            
        Returns:
            Tuple[int, int]: Hat position (x, y) or (0, 0) if not found
        """
        state = self.read_state(device_id)
        if state and hat_id in state.hats:
            return state.hats[hat_id]
        return (0, 0)
    
    def print_state(self, device_id: int = 0):
        """
        Print the current state of a joystick device for debugging.
        
        Args:
            device_id: The ID of the joystick device to print
        """
        state = self.read_state(device_id)
        if state:
            print(f"\nJoystick {device_id} State (t={state.timestamp:.3f}):")
            print(f"  Axes: {state.axes}")
            print(f"  Buttons: {state.buttons}")
            print(f"  Hats: {state.hats}")
            print(f"  Connected: {state.connected}")
        else:
            print(f"Joystick {device_id} not found or not connected")
    
    def cleanup(self):
        """Clean up pygame and disconnect all devices"""
        try:
            with self._lock:
                # Disconnect all devices
                for device_id in list(self.joysticks.keys()):
                    self.disconnect_device(device_id)
                
                # Quit pygame
                pygame.quit()
                print("Joystick interface cleaned up")
                
        except Exception as e:
            print(f"Error during cleanup: {e}")


def main():
    """Example usage of the joystick interface"""
    joystick = JoystickInterface()
    
    try:
        # Connect to the first available joystick
        if joystick.connect_device(0):
            print("Joystick connected successfully!")
            
            # Print device info
            info = joystick.get_device_info(0)
            if info:
                print(f"Device info: {info}")
            
            # Read and print state for a few seconds
            print("\nReading joystick state for 5 seconds...")
            start_time = time.time()
            
            while time.time() - start_time < 5:
                joystick.print_state(0)
                time.sleep(0.1)  # Read at 10Hz
            
            # Disconnect
            joystick.disconnect_device(0)
        else:
            print("No joystick found or connection failed")
    
    finally:
        joystick.cleanup()


if __name__ == "__main__":
    main()
