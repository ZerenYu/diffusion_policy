#!/usr/bin/env python3
"""
Test runner for Orbbec camera tests
"""

import sys
import os
import time

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_single_orbbec import test_single_orbbec
from test_intrinsics import test_intrinsics

def run_all_tests():
    """Run all Orbbec tests"""
    print("=" * 50)
    print("Running Orbbec Camera Tests")
    print("=" * 50)
    
    # Test 1: Basic functionality
    print("\n1. Testing basic SingleOrbbec functionality...")
    try:
        test_single_orbbec()
        print("✓ Basic functionality test passed")
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
    
    time.sleep(2)  # Give some time between tests
    
    # Test 2: Intrinsics extraction
    print("\n2. Testing intrinsics extraction...")
    try:
        test_intrinsics()
        print("✓ Intrinsics extraction test passed")
    except Exception as e:
        print(f"✗ Intrinsics extraction test failed: {e}")
    
    print("\n" + "=" * 50)
    print("All tests completed!")
    print("=" * 50)

if __name__ == "__main__":
    run_all_tests() 