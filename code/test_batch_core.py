#!/usr/bin/env python3
"""
Simple test for BATCH processing - Core only
Tests the CoreCommunicator BATCH functionality directly
"""
import os
import sys
import numpy as np
from PIL import Image
import tempfile
import shutil
import time
from core import HoloTrackerCore
from core_communicator import CoreCommunicator, CommandType, Command

def create_test_hologram(filename, size=(256, 256)):
    """Create a simple synthetic hologram for testing"""
    data = np.random.randint(100, 200, size, dtype=np.uint8)
    
    # Add some circular patterns
    x = np.arange(size[0])
    y = np.arange(size[1])
    X, Y = np.meshgrid(x, y)
    
    # Add 2 bright spots
    for cx, cy in [(80, 80), (180, 150)]:
        mask = ((X - cx)**2 + (Y - cy)**2) < 400  # radius ~20
        data[mask] = 250
    
    img = Image.fromarray(data)
    img.save(filename)
    return filename

def test_core_batch_functionality():
    """Test BATCH processing at the Core level"""
    print("🚀 Testing Core BATCH functionality...")
    
    # Create test directory
    test_dir = tempfile.mkdtemp(prefix="batch_core_test_")
    print(f"📁 Test directory: {test_dir}")
    
    try:
        # Create test holograms
        hologram_files = []
        for i in range(3):
            filename = f"hologram_{i:03d}.tiff"
            filepath = os.path.join(test_dir, filename)
            create_test_hologram(filepath)
            hologram_files.append(filename)
            print(f"✅ Created: {filename}")
        
        # Initialize Core and CoreCommunicator
        core = HoloTrackerCore()
        
        # Set basic parameters
        core.set_parameter("holo_size_x", 256)
        core.set_parameter("holo_size_y", 256)
        core.set_parameter("wavelength", 532e-9)
        core.set_parameter("optical_index", 1.33)
        core.set_parameter("magnification", 63.0)
        core.set_parameter("pixel_size", 6.45e-6)
        core.set_parameter("remove_mean", False)
        core.set_parameter("nb_plan_reconstruit", 10)
        
        print("✅ Core parameters set")
        
        # Start CoreCommunicator
        core_comm = CoreCommunicator(core)
        core_comm.start()
        print("✅ CoreCommunicator started")
        
        # Test 1: ENTER_BATCH_MODE
        print("\n🔄 Test 1: ENTER_BATCH_MODE")
        command_data = {
            'parameters': core.parameters,
            'directory': test_dir
        }
        core_comm.send_command(CommandType.ENTER_BATCH_MODE, command_data)
        
        # Wait and check result
        time.sleep(2)
        result = core_comm.get_result(timeout=1)
        if result:
            print(f"✅ ENTER_BATCH_MODE result: {result.success}")
            if result.success:
                csv_path = result.data.get('csv_path', '')
                print(f"📄 CSV created: {os.path.basename(csv_path)}")
        else:
            print("❌ No result received for ENTER_BATCH_MODE")
        
        # Test 2: Process holograms in batch
        print("\n🔄 Test 2: Processing holograms")
        for filename in hologram_files:
            print(f"Processing {filename}...")
            core_comm.send_command(
                CommandType.PROCESS_HOLOGRAM_BATCH,
                {'directory': test_dir, 'filename': filename}
            )
            
            time.sleep(1)
            result = core_comm.get_result(timeout=1)
            if result and result.success:
                batch_info = result.data.get('batch_info', {})
                hologram_num = batch_info.get('hologram_number', '?')
                count = result.data.get('count', 0)
                print(f"✅ Hologram #{hologram_num} processed - Found {count} objects")
            else:
                print(f"❌ Failed to process {filename}")
        
        # Test 3: EXIT_BATCH_MODE
        print("\n🔄 Test 3: EXIT_BATCH_MODE")
        core_comm.send_command(CommandType.EXIT_BATCH_MODE, {})
        
        time.sleep(1)
        result = core_comm.get_result(timeout=1)
        if result:
            print(f"✅ EXIT_BATCH_MODE result: {result.success}")
        
        # Check CSV file
        csv_files = [f for f in os.listdir(test_dir) if f.startswith("RESULT_") and f.endswith(".csv")]
        if csv_files:
            csv_path = os.path.join(test_dir, csv_files[0])
            print(f"\n📊 CSV file found: {csv_files[0]}")
            
            if os.path.exists(csv_path):
                with open(csv_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    lines = content.split('\n')
                    print(f"📄 CSV has {len(lines)} lines:")
                    for i, line in enumerate(lines):
                        print(f"  {i}: {line}")
        else:
            print("❌ No CSV file found")
        
        # Cleanup
        core_comm.stop()
        print("\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Clean up
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
            print(f"🧹 Cleaned up: {test_dir}")

if __name__ == "__main__":
    test_core_batch_functionality()
