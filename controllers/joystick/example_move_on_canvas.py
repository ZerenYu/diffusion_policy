#!/usr/bin/env python3
"""
Example of using joystick control to move a dot on a canvas.
This script demonstrates how to use the left stick axes to control
the position of a dot on a pygame canvas.
"""

import pygame
import sys
import os
import time
import math

# Add the parent directory to the path to import joystick_control
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from joystick_control import JoystickInterface, JoystickAxis


class CanvasDotController:
    """Controls a dot on a canvas using joystick input"""
    
    def __init__(self, width=800, height=600, dot_radius=10, speed=5.0):
        """
        Initialize the canvas controller.
        
        Args:
            width: Canvas width in pixels
            height: Canvas height in pixels
            dot_radius: Radius of the controlled dot
            speed: Movement speed multiplier
        """
        self.width = width
        self.height = height
        self.dot_radius = dot_radius
        self.speed = speed
        
        # Initialize pygame display
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Joystick Dot Controller")
        
        # Initialize joystick interface
        self.joystick = JoystickInterface()
        
        # Dot position (start at center)
        self.dot_x = width // 2
        self.dot_y = height // 2
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.GREEN = (0, 255, 0)
        
        # Control settings
        self.dead_zone = 0.1  # Ignore small joystick movements
        self.max_speed = 10.0  # Maximum movement speed
        
        # Status info
        self.font = pygame.font.Font(None, 24)
        self.clock = pygame.time.Clock()
        
    def connect_joystick(self):
        """Connect to the first available joystick"""
        if self.joystick.connect_device(0):
            print("Joystick connected successfully!")
            info = self.joystick.get_device_info(0)
            if info:
                print(f"Device: {info['name']}")
                print(f"Axes: {info['num_axes']}, Buttons: {info['num_buttons']}")
            return True
        else:
            print("No joystick found. Please connect a joystick and restart.")
            return False
    
    def handle_joystick_input(self):
        """Process joystick input to move the dot"""
        state = self.joystick.read_state(0)
        if not state:
            return
        
        # Get left stick values (axes 0 and 1)
        left_x = state.axes.get(JoystickAxis.LEFT_X.value, 0.0)
        left_y = state.axes.get(JoystickAxis.LEFT_Y.value, 0.0)
        
        # Apply dead zone
        if abs(left_x) < self.dead_zone:
            left_x = 0.0
        if abs(left_y) < self.dead_zone:
            left_y = 0.0
        
        # Calculate movement
        if abs(left_x) > 0 or abs(left_y) > 0:
            # Calculate movement speed based on joystick magnitude
            magnitude = math.sqrt(left_x**2 + left_y**2)
            speed = min(magnitude * self.speed, self.max_speed)
            
            # Calculate movement vector
            dx = left_x * speed
            dy = left_y * speed
            
            # Update dot position
            new_x = self.dot_x + dx
            new_y = self.dot_y + dy
            
            # Keep dot within canvas bounds
            self.dot_x = max(self.dot_radius, min(self.width - self.dot_radius, new_x))
            self.dot_y = max(self.dot_radius, min(self.height - self.dot_radius, new_y))
    
    def draw_canvas(self):
        """Draw the canvas and dot"""
        # Clear screen
        self.screen.fill(self.BLACK)
        
        # Draw grid lines for reference
        self.draw_grid()
        
        # Draw the controlled dot
        pygame.draw.circle(self.screen, self.RED, (int(self.dot_x), int(self.dot_y)), self.dot_radius)
        
        # Draw center crosshair
        center_x, center_y = self.width // 2, self.height // 2
        pygame.draw.line(self.screen, self.GREEN, (center_x - 20, center_y), (center_x + 20, center_y), 2)
        pygame.draw.line(self.screen, self.GREEN, (center_x, center_y - 20), (center_x, center_y + 20), 2)
        
        # Draw status information
        self.draw_status()
        
        # Update display
        pygame.display.flip()
    
    def draw_grid(self):
        """Draw a grid on the canvas for reference"""
        grid_size = 50
        grid_color = (50, 50, 50)
        
        # Vertical lines
        for x in range(0, self.width, grid_size):
            pygame.draw.line(self.screen, grid_color, (x, 0), (x, self.height))
        
        # Horizontal lines
        for y in range(0, self.height, grid_size):
            pygame.draw.line(self.screen, grid_color, (0, y), (self.width, y))
    
    def draw_status(self):
        """Draw status information on screen"""
        # Position info
        pos_text = f"Position: ({int(self.dot_x)}, {int(self.dot_y)})"
        pos_surface = self.font.render(pos_text, True, self.WHITE)
        self.screen.blit(pos_surface, (10, 10))
        
        # Instructions
        instructions = [
            "Use LEFT STICK to move the red dot",
            "Press ESC to exit",
            "Press SPACE to reset position"
        ]
        
        for i, instruction in enumerate(instructions):
            text_surface = self.font.render(instruction, True, self.WHITE)
            self.screen.blit(text_surface, (10, self.height - 80 + i * 25))
    
    def reset_dot_position(self):
        """Reset dot to center of canvas"""
        self.dot_x = self.width // 2
        self.dot_y = self.height // 2
        print("Dot position reset to center")
    
    def handle_keyboard_events(self):
        """Handle keyboard events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_SPACE:
                    self.reset_dot_position()
        return True
    
    def run(self):
        """Main run loop"""
        if not self.connect_joystick():
            return
        
        print("Canvas dot controller started!")
        print("Use the left stick to move the red dot")
        print("Press ESC to exit, SPACE to reset position")
        
        running = True
        while running:
            # Handle keyboard events
            running = self.handle_keyboard_events()
            
            # Handle joystick input
            self.handle_joystick_input()
            
            # Draw everything
            self.draw_canvas()
            
            # Control frame rate
            self.clock.tick(60)
        
        # Cleanup
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.joystick.cleanup()
        pygame.quit()
        print("Canvas controller cleaned up")


def main():
    """Main function"""
    print("Joystick Canvas Dot Controller")
    print("==============================")
    print("This example demonstrates joystick control of a dot on a canvas.")
    print("Connect a joystick and use the left stick to move the red dot.")
    
    try:
        controller = CanvasDotController()
        controller.run()
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main() 