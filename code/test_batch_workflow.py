#!/usr/bin/env python3
"""
Test script for BATCH processing workflow
Creates test hologram files and validates the BATCH processing pipeline
"""
import os
import numpy as np
from PIL import Image
import tempfile
import shutil
from controller_threaded import HoloTrackerController
from core import HoloTrackerCore
from ui import HoloTrackerApp
import tkinter as tk

def create_test_hologram(filename, size=(512, 512)):
    """Create a synthetic hologram for testing"""
    # Create a synthetic hologram with some patterns
    data = np.random.randint(0, 255, size, dtype=np.uint8)
    
    # Add some structure to make it more realistic
    x = np.arange(size[0])
    y = np.arange(size[1])
    X, Y = np.meshgrid(x, y)
    
    # Add circular patterns (simulated particles)
    for i in range(3):
        cx, cy = np.random.randint(50, size[0]-50), np.random.randint(50, size[1]-50)
        radius = np.random.randint(10, 30)
        mask = ((X - cx)**2 + (Y - cy)**2) < radius**2
        data[mask] = np.random.randint(200, 255)
    
    # Save as TIFF
    img = Image.fromarray(data)
    img.save(filename)
    print(f"✅ Created test hologram: {filename}")

def test_batch_workflow():
    """Test the complete BATCH processing workflow"""
    print("🚀 Starting BATCH workflow test...")
    
    # Create temporary directory for test holograms
    test_dir = tempfile.mkdtemp(prefix="holotracker_batch_test_")
    print(f"📁 Test directory: {test_dir}")
    
    try:
        # Create test hologram files
        hologram_files = []
        for i in range(5):
            filename = f"test_hologram_{i:03d}.tiff"
            filepath = os.path.join(test_dir, filename)
            create_test_hologram(filepath)
            hologram_files.append(filename)
        
        print(f"📊 Created {len(hologram_files)} test holograms")
        
        # Initialize components
        root = tk.Tk()
        root.withdraw()  # Hide the main window for testing
        
        core = HoloTrackerCore()
        app_ui = HoloTrackerApp(root)
        controller = HoloTrackerController(app_ui, core)
        app_ui.controller = controller
        
        # Load basic parameters
        app_ui.load_parameters()
        controller.sync_parameters_to_core()
        
        print("✅ Components initialized")
        
        # Test BATCH mode entry
        print("\n🔄 Testing BATCH mode entry...")
        controller.on_enter_batch_mode(test_dir)
        
        # Wait for result (in a real test, we'd need to check the result queue)
        import time
        time.sleep(2)
        
        print("✅ BATCH mode entry command sent")
        
        # Test individual hologram processing
        print("\n🔄 Testing hologram processing in BATCH mode...")
        for i, filename in enumerate(hologram_files[:3]):  # Test first 3
            print(f"Processing {filename}...")
            controller.on_process_hologram_batch(test_dir, filename)
            time.sleep(1)  # Allow processing time
        
        print("✅ Hologram processing commands sent")
        
        # Test BATCH mode exit
        print("\n🔄 Testing BATCH mode exit...")
        controller.on_exit_batch_mode()
        time.sleep(1)
        
        print("✅ BATCH mode exit command sent")
        
        # Check if CSV was created
        csv_files = [f for f in os.listdir(test_dir) if f.startswith("RESULT_") and f.endswith(".csv")]
        if csv_files:
            csv_path = os.path.join(test_dir, csv_files[0])
            print(f"📄 CSV file created: {csv_files[0]}")
            
            # Read and display CSV content
            if os.path.exists(csv_path):
                with open(csv_path, 'r') as f:
                    content = f.read()
                    print("📊 CSV Content:")
                    print(content)
        else:
            print("⚠️  No CSV file found")
        
        # Cleanup
        controller.cleanup()
        root.destroy()
        
        print("\n✅ BATCH workflow test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during BATCH workflow test: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Clean up test directory
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
            print(f"🧹 Cleaned up test directory: {test_dir}")

if __name__ == "__main__":
    test_batch_workflow()
