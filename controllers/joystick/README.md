# Joystick Control Interface

A basic joystick interface implementation using pygame for the teleopt system. This interface provides device connection and state reading capabilities for joystick control applications.

## Features

- **Device Connection**: Connect to joystick devices by ID
- **State Reading**: Read current joystick state including axes, buttons, and hats
- **Thread-Safe**: Thread-safe operations for concurrent access
- **Dead Zone Handling**: Built-in dead zone to prevent drift
- **Multiple Device Support**: Support for multiple connected joysticks
- **Cross-Platform**: Works on Linux, Windows, and macOS

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from joystick_control import JoystickInterface

# Initialize the joystick interface
joystick = JoystickInterface()

# Connect to the first available joystick
if joystick.connect_device(0):
    # Read the current state
    state = joystick.read_state(0)
    
    if state:
        # Access axis values (-1.0 to 1.0)
        left_x = state.axes.get(0, 0.0)  # Left stick X
        left_y = state.axes.get(1, 0.0)  # Left stick Y
        
        # Access button states (True/False)
        a_button = state.buttons.get(0, False)  # A button
        
        # Access hat states ((x, y) position)
        hat_state = state.hats.get(0, (0, 0))  # First hat
        
        print(f"Left stick: ({left_x:.2f}, {left_y:.2f})")
        print(f"A button: {a_button}")
        print(f"Hat: {hat_state}")
    
    # Disconnect when done
    joystick.disconnect_device(0)

# Clean up
joystick.cleanup()
```

### Continuous Reading

```python
import time
from joystick_control import JoystickInterface

joystick = JoystickInterface()

try:
    if joystick.connect_device(0):
        print("Joystick connected! Press Ctrl+C to exit.")
        
        while True:
            state = joystick.read_state(0)
            
            if state:
                # Process joystick input
                left_x = state.axes.get(0, 0.0)
                left_y = state.axes.get(1, 0.0)
                
                if abs(left_x) > 0.1 or abs(left_y) > 0.1:
                    print(f"Movement: ({left_x:.2f}, {left_y:.2f})")
            
            time.sleep(0.1)  # 10Hz reading rate
            
except KeyboardInterrupt:
    print("Exiting...")
finally:
    joystick.cleanup()
```

### Robot Control Example

```python
from joystick_control import JoystickInterface, JoystickButton, JoystickAxis

joystick = JoystickInterface()

if joystick.connect_device(0):
    while True:
        state = joystick.read_state(0)
        
        if state:
            # Get movement controls
            left_x = state.axes.get(JoystickAxis.LEFT_X.value, 0.0)
            left_y = state.axes.get(JoystickAxis.LEFT_Y.value, 0.0)
            
            # Get rotation controls
            right_x = state.axes.get(JoystickAxis.RIGHT_X.value, 0.0)
            right_y = state.axes.get(JoystickAxis.RIGHT_Y.value, 0.0)
            
            # Get trigger controls
            left_trigger = state.axes.get(JoystickAxis.LEFT_TRIGGER.value, 0.0)
            right_trigger = state.axes.get(JoystickAxis.RIGHT_TRIGGER.value, 0.0)
            
            # Check buttons
            a_button = state.buttons.get(JoystickButton.A.value, False)
            b_button = state.buttons.get(JoystickButton.B.value, False)
            
            # Process robot commands
            if abs(left_x) > 0.1 or abs(left_y) > 0.1:
                # Send movement command to robot
                pass
            
            if abs(right_x) > 0.1 or abs(right_y) > 0.1:
                # Send rotation command to robot
                pass
            
            if a_button:
                # Emergency stop
                pass
            
            if b_button:
                # Reset position
                pass
```

## API Reference

### JoystickInterface

Main class for joystick control.

#### Methods

- `connect_device(device_id: int) -> bool`: Connect to a joystick device
- `disconnect_device(device_id: int) -> bool`: Disconnect a joystick device
- `read_state(device_id: int) -> Optional[JoystickState]`: Read current joystick state
- `get_connected_devices() -> List[int]`: Get list of connected device IDs
- `get_device_info(device_id: int) -> Optional[Dict]`: Get device information
- `get_axis_value(device_id: int, axis_id: int) -> float`: Get specific axis value
- `get_button_state(device_id: int, button_id: int) -> bool`: Get specific button state
- `get_hat_state(device_id: int, hat_id: int) -> Tuple[int, int]`: Get specific hat state
- `print_state(device_id: int)`: Print current state for debugging
- `cleanup()`: Clean up pygame and disconnect all devices

### JoystickState

Data class representing joystick state.

#### Attributes

- `axes: Dict[int, float]`: Axis values (-1.0 to 1.0)
- `buttons: Dict[int, bool]`: Button pressed states
- `hats: Dict[int, Tuple[int, int]]`: Hat positions
- `timestamp: float`: Timestamp of the reading
- `connected: bool`: Connection status

### JoystickButton

Enum for standard button mappings.

### JoystickAxis

Enum for standard axis mappings.

## Examples

Run the example script to see different usage patterns:

```bash
python example_usage.py
```

This will show you:
1. Basic control example
2. Robot control example
3. Device discovery example

## Troubleshooting

### No Joystick Found

1. Make sure your joystick is connected and recognized by your operating system
2. On Linux, you may need to install joystick drivers
3. Check if pygame can detect your joystick:
   ```python
   import pygame
   pygame.init()
   pygame.joystick.init()
   print(f"Found {pygame.joystick.get_count()} joystick(s)")
   ```

### Permission Issues (Linux)

If you get permission errors on Linux, you may need to add your user to the input group:

```bash
sudo usermod -a -G input $USER
```

Then log out and log back in.

### Windows Issues

On Windows, make sure you have the latest DirectX runtime installed.

## Integration with Teleopt

This joystick interface is designed to integrate with the teleopt system:

1. **Controllers**: Use the joystick interface in your controller implementations
2. **Executors**: Pass joystick commands to executors for robot control
3. **Configuration**: Extend the interface to support configuration files
4. **ROS2 Integration**: Add ROS2 topics and services for joystick control

## Future Enhancements

- Configuration file support
- Calibration tools
- Force feedback support
- Network joystick support
- ROS2 integration
- Web-based interface 