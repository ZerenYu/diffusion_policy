# Orbbec Camera Tests

This directory contains tests for the SingleOrbbec implementation.

## Test Files

- **`test_single_orbbec.py`**: Basic functionality test for SingleOrbbec class
- **`test_intrinsics.py`**: Test for intrinsics extraction from frames
- **`single_orbbec_example.py`**: Example usage of SingleOrbbec class
- **`run_tests.py`**: Test runner that executes all tests

## Running Tests

### Individual Tests

```bash
# Run basic functionality test
python test_single_orbbec.py

# Run intrinsics extraction test
python test_intrinsics.py

# Run example
python single_orbbec_example.py
```

### All Tests

```bash
# Run all tests
python run_tests.py
```

## Prerequisites

1. **Orbbec Camera**: Make sure you have an Orbbec camera connected
2. **pyorbbecsdk**: Install the Orbbec SDK
3. **Dependencies**: Ensure all required packages are installed

## Test Requirements

- Connected Orbbec camera
- Proper permissions to access the camera
- Sufficient disk space for video recording tests

## Expected Output

The tests should:
1. Detect connected Orbbec devices
2. Start the camera successfully
3. Extract camera intrinsics
4. Capture color and depth frames
5. Test video recording functionality
6. Clean up resources properly

## Troubleshooting

If tests fail:
1. Check that your Orbbec camera is properly connected
2. Verify that pyorbbecsdk is installed and working
3. Ensure you have proper permissions to access the camera
4. Check the console output for specific error messages 