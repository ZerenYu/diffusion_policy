import argparse
import serial
import time
import struct
from typing import List, Tuple, Optional

class InspireGripper:
    """
    A Python class to control the Inspire Robots EG2-4X2 Servo Electric Gripper
    using the regular serial communication protocol described in the user manual.
    """

    def __init__(self, port: str, gripper_id: int = 1, baudrate: int = 115200):
        """
        Initializes the connection to the gripper.

        Args:
            port (str): The serial port name (e.g., 'COM3' on Windows,
                        '/dev/ttyUSB0' on Linux).
            gripper_id (int): The ID of the gripper, from 1 to 254.
                              The factory default is 1.
            baudrate (int): The communication baud rate. The manual's Modbus section
                            suggests 115200 is a standard rate.
        """
        if not (1 <= gripper_id <= 254):
            raise ValueError("Gripper ID must be between 1 and 254.")

        self.port = port
        self.gripper_id = gripper_id
        self.baudrate = baudrate
        self.ser = None # Initialize ser to None
        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS,
                timeout=1.0  # Set a 1-second read timeout
            )
            print(f"Successfully connected to gripper on {self.port} at {self.baudrate} baud.")
        except serial.SerialException as e:
            print(f"Error: Could not open serial port {self.port}. {e}")
            raise

    def _calculate_checksum(self, data: bytes) -> int:
        """
        Calculates the checksum for a given data frame.
        The checksum is the low byte of the sum of all bytes from the ID to the data.

        """
        return sum(data) & 0xFF

    def _send_command(self, command_id: int, data: bytes = b'') -> Optional[bytes]:
        """
        Constructs a frame, sends it to the gripper, and waits for a response.

        Args:
            command_id (int): The command instruction number (CMD).
            data (bytes): The data payload for the command.

        Returns:
            Optional[bytes]: The data part of the response frame, or None if no
                             response is expected or an error occurs.
        """
        if not self.ser or not self.ser.is_open:
            print("Error: Serial port not open. Cannot send command.")
            return None

        # Instruction Frame Structure: Header(2) + ID(1) + Len(1) + CMD(1) + Data(n) + Checksum(1)
        header = b'\xEB\x90'
        id_byte = self.gripper_id.to_bytes(1, 'little')

        # Data length = command_id length (1) + data payload length
        len_byte = (1 + len(data)).to_bytes(1, 'little')
        cmd_byte = command_id.to_bytes(1, 'little')

        # Frame part for checksum calculation
        frame_for_checksum = id_byte + len_byte + cmd_byte + data
        checksum = self._calculate_checksum(frame_for_checksum).to_bytes(1, 'little')

        # Final command frame
        full_frame = header + frame_for_checksum + checksum

        try:
            self.ser.write(full_frame)
        except serial.SerialException as e:
            print(f"Error writing to serial port: {e}")
            return None

        # Broadcast ID 255 does not return a response
        if self.gripper_id == 255:
            return None

        return self._read_response(command_id)

    def _read_response(self, sent_command_id: int) -> Optional[bytes]:
        """
        Reads and validates the response frame from the gripper.

        Returns:
            The data part of the response frame.
        """
        if not self.ser or not self.ser.is_open:
            print("Error: Serial port not open. Cannot read response.")
            return None

        try:
            # Response Frame Structure: Header(2) + ID(1) + Len(1) + CMD(1) + Data(n) + Checksum(1)
            response_header = self.ser.read(2)
            if response_header != b'\xEE\x16':
                print(f"Warning: Invalid response header: {response_header.hex().upper()}")
                return None

            # Read the rest of the frame base
            id_res = self.ser.read(1)
            len_res = self.ser.read(1)
            cmd_res = self.ser.read(1)

            if not all([id_res, len_res, cmd_res]):
                print("Warning: Timed out waiting for response or incomplete response.")
                return None

            # print(f"Response ID: {id_res.hex().upper()}, Length: {len_res.hex().upper()}, Command: {cmd_res.hex().upper()}")

            data_len = int.from_bytes(len_res, 'little') - 1
            if data_len < 0:
                print(f"Warning: Invalid data length received: {data_len}. Expected at least 1 for CMD.")
                return None

            response_data = self.ser.read(data_len)

            checksum_res = self.ser.read(1)

            # Validate checksum
            frame_for_checksum = id_res + len_res + cmd_res + response_data
            calculated_checksum = self._calculate_checksum(frame_for_checksum).to_bytes(1, 'little')
            if checksum_res != calculated_checksum:
                print(f"Warning: Response checksum mismatch. Got {checksum_res.hex().upper()}, expected {calculated_checksum.hex().upper()}")
                return None

            # For write commands, the first byte of data is a status flag.
            # 0x01 means success, 0x55 means failure
            # We check this in the specific functions.

            return response_data
        except serial.SerialException as e:
            print(f"Error reading from serial port: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during response reading: {e}")
            return None

    # --- High-Level Control Functions ---

    def grip(self, speed: int = 500, force_threshold: int = 100) -> bool:
        """
        Commands the gripper to close with a specific speed and force limit.
        Stops when the force threshold is exceeded.

        Args:
            speed (int): Movement speed, from 1 to 1000.
            force_threshold (int): Force limit, from 50 to 1000 (e.g., 100 for 100g).

        Returns:
            bool: True if the command was acknowledged successfully, False otherwise.
        """
        if not (1 <= speed <= 1000):
            print("Error: Speed must be between 1 and 1000.")
            return False
        if not (50 <= force_threshold <= 1000):
            print("Error: Force threshold must be between 50 and 1000.")
            return False

        command_id = 0x10  # CMD_MC_MOVE_CATCH_XG
        speed_bytes = struct.pack('<H', speed)  # Little-endian, 2 bytes
        force_bytes = struct.pack('<H', force_threshold) # Little-endian, 2 bytes
        data = speed_bytes + force_bytes

        response = self._send_command(command_id, data)
        if response and response[0] == 0x01:
            print(f"Gripper commanded to grip with speed {speed} and force {force_threshold}.")
            return True
        print(f"Gripper grip command failed. Response: {response.hex() if response else 'None'}")
        return False

    def grip_continuous(self, speed: int = 500, force_threshold: int = 100) -> bool:
        """
        Commands the gripper to close and maintain the grasping force.
        If the object slips and the force drops, the gripper will re-grip.

        Args:
            speed (int): Movement speed, from 1 to 1000.
            force_threshold (int): Force limit, from 50 to 1000 (e.g., 100 for 100g).

        Returns:
            bool: True if the command was acknowledged successfully, False otherwise.
        """
        if not (1 <= speed <= 1000):
            print("Error: Speed must be between 1 and 1000.")
            return False
        if not (50 <= force_threshold <= 1000):
            print("Error: Force threshold must be between 50 and 1000.")
            return False

        command_id = 0x18  # CMD_MC_MOVE_CATCH2_XG
        speed_bytes = struct.pack('<H', speed)
        force_bytes = struct.pack('<H', force_threshold)
        data = speed_bytes + force_bytes

        response = self._send_command(command_id, data)
        if response and response[0] == 0x01:
            print(f"Gripper commanded to continuous grip with speed {speed} and force {force_threshold}.")
            return True
        print(f"Gripper continuous grip command failed. Response: {response.hex() if response else 'None'}")
        return False

    def release(self, speed: int = 500) -> bool:
        """
        Commands the gripper to open to its maximum position at a given speed.

        Args:
            speed (int): Movement speed, from 1 to 1000.

        Returns:
            bool: True if the command was acknowledged successfully, False otherwise.
        """
        if not (1 <= speed <= 1000):
            print("Error: Speed must be between 1 and 1000.")
            return False

        command_id = 0x11  # CMD_MC_MOVE_RELEASE
        speed_bytes = struct.pack('<H', speed)

        response = self._send_command(command_id, speed_bytes)
        if response and response[0] == 0x01:
            print(f"Gripper commanded to release with speed {speed}.")
            return True
        print(f"Gripper release command failed. Response: {response.hex() if response else 'None'}")
        return False

    def move_to(self, position: int) -> bool:
        """
        Moves the gripper to a specified opening position.
        The value is between 0 (closed) and 1000 (fully open, 70mm).

        Args:
            position (int): Target opening value, 1-1000.

        Returns:
            bool: True if the command was acknowledged successfully, False otherwise.
        """
        if not (0 <= position <= 1000):
            raise ValueError("Position must be between 1 and 1000.")

        command_id = 0x54 # CMD_MC_SEEKPOS
        pos_bytes = struct.pack('<H', position)

        response = self._send_command(command_id, pos_bytes)
        if response and response[0] == 0x01:
            print(f"Gripper commanded to move to position {position}.")
            return True
        print(f"Gripper move_to command failed. Response: {response.hex() if response else 'None'}")
        return False

    def stop(self) -> bool:
        """
        Commands the gripper to stop all motion immediately.

        Returns:
            bool: True if the command was acknowledged successfully, False otherwise.
        """
        command_id = 0x16 # CMD_MC_MOVE_STOPHERE
        response = self._send_command(command_id)
        if response and response[0] == 0x01:
            print("Gripper commanded to stop.")
            return True
        print(f"Gripper stop command failed. Response: {response.hex() if response else 'None'}")
        return False

    def read_position(self) -> Optional[int]:
        """
        Reads the current opening position of the gripper.

        Returns:
            Optional[int]: The current position (1-1000), or None on failure.
        """
        command_id = 0xD9  # CMD_MC_READ_ACTPOS
        response = self._send_command(command_id)
        if response and len(response) == 2:
            # Response data is 2 bytes, little-endian
            position = struct.unpack('<H', response)[0]
            print(f"Current gripper position: {position}")
            return position
        print(f"Failed to read position. Response: {response.hex() if response else 'None'}")
        return None

    def read_status(self) -> Optional[dict]:
        """
        Reads the detailed running status of the gripper.
        Includes status, fault code, temperature, position, and force setting.

        Returns:
            A dictionary containing the status information, or None on failure.
        """
        command_id = 0x41 # CMD_MC_READ_EG_RUNSTATE
        response = self._send_command(command_id)
        if response and len(response) == 7:
            # Parse the 8-byte response data
            run_status_code, fault_code, temp, pos, force = struct.unpack('<BBBH H', response)

            # Decode status codes
            status_map = {
                0x01: "Open and idle",
                0x02: "Closed and idle",
                0x03: "Stopped and idle",
                0x04: "Closing",
                0x05: "Opening",
                0x06: "Stopped by force limit during grip"
            }

            # Decode fault bits
            faults = {
                "stall": bool(fault_code & 0b00000001),
                "over_temperature": bool(fault_code & 0b00000010),
                "over_current": bool(fault_code & 0b00000100),
                "driver_fault": bool(fault_code & 0b00001000),
                "comm_fault": bool(fault_code & 0b00010000),
            }

            status_info = {
                "status_code": run_status_code,
                "status_text": status_map.get(run_status_code, "Unknown"),
                "fault_code": fault_code,
                "faults": faults,
                "temperature_celsius": temp, # Temperature is in Celsius
                "current_position": pos,
                "force_setting_g": force # Force setting is in grams
            }
            # print("Gripper Status:")
            # for key, value in status_info.items():
            #     print(f"  {key}: {value}")
            return status_info
        print(f"Failed to read status. Response: {response.hex() if response else 'None'}")
        return None

    def clear_fault(self) -> bool:
        """
        Clears resettable faults (stall, over-current, driver, comms).
        Over-temperature faults clear automatically when cooled.

        Returns:
            bool: True if the command was acknowledged successfully, False otherwise.
        """
        command_id = 0x17 # CMD_MC_ERROR_CLR
        response = self._send_command(command_id)
        if response and response[0] == 0x01:
            print("Gripper faults cleared.")
            return True
        print(f"Gripper clear_fault command failed. Response: {response.hex() if response else 'None'}")
        return False

    def set_id(self, new_id: int) -> bool:
        """
        Sets a new ID for the gripper.

        Args:
            new_id (int): The new gripper ID (1-254).

        Returns:
            bool: True if the command was acknowledged successfully. Note that after
                  this, the gripper will only respond to the new ID.
        """
        if not (1 <= new_id <= 254):
            raise ValueError("New gripper ID must be between 1 and 254.")

        command_id = 0x04 # CMD_MC_PARA_ID_SET
        data = new_id.to_bytes(1, 'little')

        response = self._send_command(command_id, data)
        if response and response[0] == 0x01:
            print(f"Gripper ID successfully changed to {new_id}. Remember to update your connection ID for future commands.")
            self.gripper_id = new_id
            return True

        print(f"Failed to set new ID. Response: {response.hex() if response else 'None'}")
        return False

    def save_parameters(self) -> bool:
        """
        Saves the current parameters (like ID, opening limits) to the gripper's
        internal flash memory so they persist after power loss.

        Note: This operation can take up to 1 second.

        Returns:
            bool: True if the command was acknowledged successfully.
        """
        command_id = 0x01 # CMD_MC_PARA_SAVE
        response = self._send_command(command_id)
        # It's good practice to wait after a save command
        time.sleep(1.0)
        if response and response[0] == 0x01:
            print("Gripper parameters saved.")
            return True
        print(f"Save parameters command failed. Response: {response.hex() if response else 'None'}")
        return False

    def close(self):
        """
        Closes the serial port connection.
        """
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("Serial port closed.")


def main():
    parser = argparse.ArgumentParser(
        description="Control Inspire Robots EG2-4X2 Servo Electric Gripper via CLI."
    )
    parser.add_argument(
        "--port",
        type=str,
        default="/dev/ttyUSB0",
        help="Serial port name (e.g., 'COM3' or '/dev/ttyUSB0')."
    )
    parser.add_argument(
        "--id",
        type=int,
        default=1,
        help="Gripper ID (1-254). Default is 1."
    )
    parser.add_argument(
        "--baudrate",
        type=int,
        default=115200,
        help="Baud rate for serial communication. Default is 115200."
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Grip command
    grip_parser = subparsers.add_parser("grip", help="Close the gripper with a specified speed and force limit.")
    grip_parser.add_argument(
        "--speed",
        type=int,
        default=500,
        help="Movement speed (1-1000). Default is 500."
    )
    grip_parser.add_argument(
        "--force",
        type=int,
        default=100,
        help="Force limit (50-1000, e.g., 100 for 100g). Default is 100."
    )

    # Grip Continuous command
    grip_continuous_parser = subparsers.add_parser("grip-continuous", help="Close the gripper and maintain grasping force.")
    grip_continuous_parser.add_argument(
        "--speed",
        type=int,
        default=500,
        help="Movement speed (1-1000). Default is 500."
    )
    grip_continuous_parser.add_argument(
        "--force",
        type=int,
        default=100,
        help="Force limit (50-1000, e.g., 100 for 100g). Default is 100."
    )

    # Release command
    release_parser = subparsers.add_parser("release", help="Open the gripper to its maximum position.")
    release_parser.add_argument(
        "--speed",
        type=int,
        default=500,
        help="Movement speed (1-1000). Default is 500."
    )

    # Move-to command
    move_to_parser = subparsers.add_parser("move-to", help="Move the gripper to a specified opening position.")
    move_to_parser.add_argument(
        "position",
        type=int,
        help="Target opening position (1-1000, 1=closed, 1000=fully open)."
    )

    # Stop command
    stop_parser = subparsers.add_parser("stop", help="Stop all gripper motion immediately.")

    # Read Position command
    read_pos_parser = subparsers.add_parser("read-position", help="Read the current opening position of the gripper.")

    # Read Status command
    read_status_parser = subparsers.add_parser("read-status", help="Read the detailed running status of the gripper.")

    # Clear Fault command
    clear_fault_parser = subparsers.add_parser("clear-fault", help="Clear resettable faults on the gripper.")

    # Set ID command
    set_id_parser = subparsers.add_parser("set-id", help="Set a new ID for the gripper.")
    set_id_parser.add_argument(
        "new_id",
        type=int,
        help="The new gripper ID (1-254)."
    )

    # Save Parameters command
    save_params_parser = subparsers.add_parser("save-parameters", help="Save current parameters to gripper's internal memory.")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    gripper = None
    try:
        gripper = InspireGripper(port=args.port, gripper_id=args.id, baudrate=args.baudrate)

        if args.command == "grip":
            gripper.grip(speed=args.speed, force_threshold=args.force)
        elif args.command == "grip-continuous":
            gripper.grip_continuous(speed=args.speed, force_threshold=args.force)
        elif args.command == "release":
            gripper.release(speed=args.speed)
        elif args.command == "move-to":
            gripper.move_to(position=args.position)
        elif args.command == "stop":
            gripper.stop()
        elif args.command == "read-position":
            gripper.read_position()
        elif args.command == "read-status":
            gripper.read_status()
        elif args.command == "clear-fault":
            gripper.clear_fault()
        elif args.command == "set-id":
            gripper.set_id(new_id=args.new_id)
        elif args.command == "save-parameters":
            gripper.save_parameters()

    except ValueError as e:
        print(f"Configuration Error: {e}")
    except serial.SerialException as e:
        print(f"Serial Communication Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if gripper:
            gripper.close()

if __name__ == "__main__":
    main()
