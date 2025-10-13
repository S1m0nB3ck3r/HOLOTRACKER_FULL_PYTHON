import os
import numpy as np
from PIL import Image
import scipy.ndimage as ndi
from scipy import fft
from traitement_holo import *
try:
    import cupy as cp
    from traitement_holo import calc_holo_moyen, read_image, projection_bool
    import propagation as propag
    import focus 
    from focus import Focus_type
    from CCL3D import CCL3D, calc_threshold, CCA_CUDA_float, CCL_filter, type_threshold, dobjet
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import pandas as pd
    CUPY_AVAILABLE = True
except ImportError as e:
    # Warning: CuPy or other dependencies not available
    cp = None
    CUPY_AVAILABLE = False

def stat_plane(data, label=""):
    """
    Debug function: compute and print statistics for a CuPy array
    If complex, computes abs() first
    """
    try:
        # Handle complex data
        if hasattr(data, 'dtype') and np.iscomplexobj(cp.asnumpy(data[:1, :1])):
            data_abs = cp.abs(data)
            complex_info = " (complex -> abs)"
        else:
            data_abs = data
            complex_info = ""
            
        # Compute statistics
        data_sum = float(cp.sum(data_abs))
        data_min = float(cp.min(data_abs))
        data_max = float(cp.max(data_abs))
        data_mean = float(cp.mean(data_abs))
        data_std = float(cp.std(data_abs))
        
        # Print statistics
        print(f"📊 STAT {label}{complex_info}: sum={data_sum:.3f}, min={data_min:.3f}, max={data_max:.3f}, mean={data_mean:.3f}, std={data_std:.3f}")
        
        return {
            'sum': data_sum,
            'min': data_min, 
            'max': data_max,
            'mean': data_mean,
            'std': data_std
        }
    except Exception as e:
        # Error computing statistics
        print(f"❌ STAT {label}: Error computing statistics: {e}")
        return None

class HoloTrackerCore:
    def __init__(self):
        self.results = {}
        
        # Memory allocation status
        self.memory_allocated = False
        self.test_mode_params = {}  # To track parameters that require reallocation
        
        # Mode tracking
        self.mode = 'IDLE'  # Possible values: 'IDLE', 'TEST', 'BATCH'
        self.batch_first_hologram_done = False
        
        # Allocated GPU variables (as in test_HoloTracker_locate.py)
        self.h_raw_holo = None
        self.h_mean_holo =None
        self.h_cleaned_holo = None
        self.d_holo = None
        self.h_filtered_holo = None
        self.d_filtered_holo = None
        self.d_fft_holo = None
        self.d_fft_holo_filtered = None
        self.d_fft_holo_propag = None
        self.d_holo_propag = None
        self.d_KERNEL = None
        self.d_FFT_KERNEL = None
        self.d_volume_module = None
        self.d_bin_volume_focus = None
        self.d_mean_holo = None
        
        # Variables for results
        self.current_threshold = None
        self.current_features = None
        
        # Variables for thresholding
        self.threshold = None
        self.last_nb_StdVar_threshold = None
        
        # Hologram and system parameters
        self.mean_hologram_image_path = ""
        self.holograms_directory = ""
        self.image_type = "BMP"
        self.wavelength = 660e-9
        self.medium_optical_index = 1.33
        self.objective_magnification = 40.0
        self.pixel_size = 5.5e-6
        self.holo_size_x = 1024
        self.holo_size_y = 1024
        self.distance_ini = 20e-6
        self.step = 0.5e-6
        self.number_of_planes = 200
        
        # Cleaning parameters
        self.remove_mean = True
        self.cleaning_type = "subtraction"
        
        # Filtering parameters
        self.high_pass = 15
        self.low_pass = 125
        
        # Focus parameters
        self.focus_type = "TENEGRAD"
        self.sum_size = 15
        
        # Thresholding and detection parameters
        self.nb_StdVar_threshold = 14.0
        self.connectivity = 26
        self.min_voxel = 0
        self.max_voxel = 0
        self.batch_threshold = "compute on 1st hologram"
        
        # Display parameters
        self.additional_display = "Centroid positions"

    def set_parameters(self, **kwargs):
        """Set multiple parameters at once"""
        for name, value in kwargs.items():
            if hasattr(self, name):
                setattr(self, name, value)
            else:
                raise AttributeError(f"Parameter '{name}' does not exist")
        return f"Updated {len(kwargs)} parameters"

    def get_parameters_dict(self):
        """Get all parameters as a dictionary"""
        param_attrs = [attr for attr in dir(self) if not attr.startswith('_') and 
                      not callable(getattr(self, attr)) and 
                      attr not in ['results', 'memory_allocated', 'test_mode_params',
                                  'h_raw_holo', 'h_mean_holo', 'h_cleaned_holo', 'd_holo',
                                  'h_filtered_holo', 'd_filtered_holo', 'd_fft_holo', 
                                  'd_volume', 'd_focus', 'd_threshold', 'd_CCL', 'd_filtered_objects', 'd_slices']]
        return {attr: getattr(self, attr) for attr in param_attrs}

    def load_mean_hologram(self):
        """Load mean hologram from TIF or NPY file"""
        if self.mean_hologram_image_path.lower().endswith('.npy'):
            # Legacy NPY format
            self.h_mean_holo = np.load(self.mean_hologram_image_path)
        else:
            # New TIF format
            mean_pil = Image.open(self.mean_hologram_image_path)
            self.h_mean_holo = np.array(mean_pil)

    def print_parameters(self):
        """Print all parameters, one parameter per line"""
        print("=== HOLOTRACKER PARAMETERS ===")
        # Print all parameter attributes (skip internal variables)
        param_attrs = [attr for attr in dir(self) if not attr.startswith('_') and 
                      not callable(getattr(self, attr)) and 
                      attr not in ['results', 'memory_allocated', 'test_mode_params',
                                  'h_raw_holo', 'h_mean_holo', 'h_cleaned_holo', 'd_holo',
                                  'h_filtered_holo', 'd_filtered_holo', 'd_fft_holo', 
                                  'd_fft_holo_filtered', 'd_fft_holo_propag', 'd_holo_propag',
                                  'd_KERNEL', 'd_FFT_KERNEL', 'd_volume_module', 'd_bin_volume_focus',
                                  'd_mean_holo', 'current_threshold', 'current_features',
                                  'threshold', 'last_nb_StdVar_threshold']]
        
        for attr in sorted(param_attrs):
            value = getattr(self, attr)
            print(f"{attr}: {value}")
        print("==============================")

    def get_display_image(self, directory, filename, display_type, plane_number=0, additional_display="None"):
        """Retourne l'image à afficher selon le type demandé avec marqueurs de détection"""
        try:
            # Helper: normalize a numpy array to uint8 for display
            def _to_uint8(arr):
                arr = np.array(arr)  # ensure numpy
                if np.iscomplexobj(arr):
                    arr = np.abs(arr)
                arr = arr.astype(np.float64)
                mx = arr.max() if arr.size else 0.0
                mn = arr.min() if arr.size else 0.0
                if mx > mn:
                    out = ((arr - mn) * 255.0 / (mx - mn)).astype(np.uint8)
                elif mx > 0:
                    out = (arr * 255.0 / mx).astype(np.uint8)
                else:
                    out = np.zeros_like(arr, dtype=np.uint8)
                return out

            # Provide a safe getter for raw hologram (prefer in-memory)
            def _get_raw_array():
                if self.h_raw_holo is not None:
                    return self.h_raw_holo.copy()
                # fallback to reading file
                raw_hologram = self.open_hologram_image(directory, filename)
                return np.array(raw_hologram.convert('L'), dtype=np.float64)

            # Common fallback behavior: print minimal notice and return raw hologram display
            def _fallback(type_name):
                # print(f"no '{type_name}' available")
                arr = _get_raw_array()
                return Image.fromarray(_to_uint8(arr))

            # RAW_HOLOGRAM: use in-memory if available
            if display_type == "RAW_HOLOGRAM":
                arr = _get_raw_array()
                stat_plane(arr, label="RAW_HOLOGRAM")
                image = Image.fromarray(_to_uint8(arr))
                return self._apply_additional_display(image, additional_display, display_type, plane_number)

            # CLEANED_HOLOGRAM
            if display_type == "CLEANED_HOLOGRAM":
                if self.memory_allocated and self.h_cleaned_holo is not None:
                    arr = self.h_cleaned_holo.copy()
                    stat_plane(arr, label="CLEANED_HOLOGRAM")
                    image = Image.fromarray(_to_uint8(arr))
                    return self._apply_additional_display(image, additional_display, display_type, plane_number)
                return _fallback("CLEANED_HOLOGRAM")

            # FILTERED_HOLOGRAM
            if display_type == "FILTERED_HOLOGRAM":
                # Try CPU version first (more efficient)
                if self.memory_allocated and self.h_filtered_holo is not None:
                    try:
                        h_filtered = self.h_filtered_holo.copy()
                        if np.iscomplexobj(h_filtered):
                            h_filtered = np.abs(h_filtered)
                        stat_plane(h_filtered, label="FILTERED_HOLOGRAM")
                        image = Image.fromarray(_to_uint8(h_filtered))
                        return self._apply_additional_display(image, additional_display, display_type, plane_number)
                    except Exception:
                        pass
                # Fallback to GPU version
                if self.memory_allocated and self.d_filtered_holo is not None:
                    try:
                        h_filtered = cp.asnumpy(self.d_filtered_holo)
                        if np.iscomplexobj(h_filtered):
                            h_filtered = np.abs(h_filtered)
                        stat_plane(h_filtered, label="FILTERED_HOLOGRAM")
                        image = Image.fromarray(_to_uint8(h_filtered))
                        return self._apply_additional_display(image, additional_display, display_type, plane_number)
                    except Exception:
                        return _fallback("FILTERED_HOLOGRAM")
                return _fallback("FILTERED_HOLOGRAM")

            # FFT_HOLOGRAM
            if display_type == "FFT_HOLOGRAM":
                if self.memory_allocated and self.d_fft_holo is not None:
                    try:
                        h_fft = cp.asnumpy(self.d_fft_holo)
                        if np.iscomplexobj(h_fft):
                            h_fft = np.abs(h_fft)
                        stat_plane(h_fft, label="FFT_HOLOGRAM")
                        # logarithmic scaling for FFT visualization
                        h_fft = np.log(h_fft + 1e-10)
                        image = Image.fromarray(_to_uint8(h_fft))
                        return self._apply_additional_display(image, additional_display, display_type, plane_number)
                    except Exception:
                        return _fallback("FFT_HOLOGRAM")
                return _fallback("FFT_HOLOGRAM")

            # FFT_FILTERED_HOLOGRAM
            if display_type == "FFT_FILTERED_HOLOGRAM":
                if self.memory_allocated and self.d_fft_holo_filtered is not None:
                    try:
                        h_fft_filtered = cp.asnumpy(self.d_fft_holo_filtered)
                        if np.iscomplexobj(h_fft_filtered):
                            h_fft_filtered = np.abs(h_fft_filtered)
                        stat_plane(h_fft_filtered, label="FFT_FILTERED_HOLOGRAM")
                        h_fft_filtered = np.log(h_fft_filtered + 1e-10)
                        image = Image.fromarray(_to_uint8(h_fft_filtered))
                        return self._apply_additional_display(image, additional_display, display_type, plane_number)
                    except Exception:
                        return _fallback("FFT_FILTERED_HOLOGRAM")
                return _fallback("FFT_FILTERED_HOLOGRAM")

            # VOLUME_PLANE_NUMBER and projections - keep original logic but use _fallback on error
            if display_type == "VOLUME_PLANE_NUMBER":
                if self.memory_allocated:
                    try:
                        volume_gpu = self.d_volume_module
                        if plane_number < volume_gpu.shape[0]:
                            plane = cp.asnumpy(volume_gpu[plane_number, :, :])
                            plane = np.abs(plane)
                            stat_plane(plane, label=f"VOLUME_PLANE_{plane_number}")
                            image = Image.fromarray(_to_uint8(plane))
                            return self._apply_additional_display(image, additional_display, display_type, plane_number)
                    except Exception:
                        pass
                return _fallback("VOLUME_PLANE_NUMBER")

            if display_type == "XY_SUM_PROJECTION":
                if self.memory_allocated:
                    try:
                        projection = cp.sum(self.d_volume_module, axis=0)
                        projection = cp.asnumpy(projection)
                        stat_plane(projection, label="XY_SUM_PROJECTION")
                        image = Image.fromarray(_to_uint8(projection))
                        return self._apply_additional_display(image, additional_display, display_type, plane_number, 'XY')
                    except Exception:
                        pass
                return _fallback("XY_SUM_PROJECTION")

            if display_type == "XZ_SUM_PROJECTION":
                if self.memory_allocated:
                    try:
                        projection = cp.sum(self.d_volume_module, axis=1)
                        projection = cp.asnumpy(projection)
                        stat_plane(projection, label="XZ_SUM_PROJECTION")
                        image = Image.fromarray(_to_uint8(projection))
                        return self._apply_additional_display(image, additional_display, display_type, plane_number, 'XZ')
                    except Exception:
                        pass
                return _fallback("XZ_SUM_PROJECTION")

            if display_type == "YZ_SUM_PROJECTION":
                if self.memory_allocated:
                    try:
                        print(f"Debug: Computing YZ_SUM_PROJECTION, volume shape: {self.d_volume_module.shape}")
                        print(f"Debug: Volume dtype: {self.d_volume_module.dtype}")
                        print(f"Debug: Testing simple axis=2 operation...")
                        
                        # Test with smaller slice first
                        test_slice = self.d_volume_module[:10, :10, :]
                        test_result = cp.sum(test_slice, axis=2)
                        print(f"Debug: Small test successful, result shape: {test_result.shape}")
                        
                        # Try CUDA operation first, fallback to CPU if it fails
                        try:
                            projection = cp.sum(self.d_volume_module, axis=2)
                            projection = cp.asnumpy(projection)
                            print("Debug: Full CUDA operation successful")
                        except Exception as cuda_error:
                            print(f"CUDA error in YZ_SUM_PROJECTION, falling back to CPU: {cuda_error}")
                            # Fallback to CPU computation
                            volume_cpu = cp.asnumpy(self.d_volume_module)
                            projection = np.sum(volume_cpu, axis=2)
                        
                        print(f"Debug: YZ projection shape: {projection.shape}")
                        stat_plane(projection, label="YZ_SUM_PROJECTION")
                        image = Image.fromarray(_to_uint8(projection))
                        return self._apply_additional_display(image, additional_display, display_type, plane_number, 'YZ')
                    except Exception as e:
                        print(f"Error in YZ_SUM_PROJECTION: {e}")
                        pass
                return _fallback("YZ_SUM_PROJECTION")

            if display_type == "XY_MAX_PROJECTION":
                if self.memory_allocated:
                    try:
                        projection = cp.max(self.d_volume_module, axis=0)
                        projection = cp.asnumpy(projection)
                        stat_plane(projection, label="XY_MAX_PROJECTION")
                        image = Image.fromarray(_to_uint8(projection))
                        return self._apply_additional_display(image, additional_display, display_type, plane_number, 'XY')
                    except Exception:
                        pass
                return _fallback("XY_MAX_PROJECTION")

            if display_type == "XZ_MAX_PROJECTION":
                if self.memory_allocated:
                    try:
                        projection = cp.max(self.d_volume_module, axis=1)
                        projection = cp.asnumpy(projection)
                        stat_plane(projection, label="XZ_MAX_PROJECTION")
                        image = Image.fromarray(_to_uint8(projection))
                        return self._apply_additional_display(image, additional_display, display_type, plane_number, 'XZ')
                    except Exception:
                        pass
                return _fallback("XZ_MAX_PROJECTION")

            if display_type == "YZ_MAX_PROJECTION":
                if self.memory_allocated:
                    try:
                        # print(f"Debug: Computing YZ_MAX_PROJECTION, volume shape: {self.d_volume_module.shape}")
                        # Try CUDA operation first, fallback to CPU if it fails
                        try:
                            projection = cp.max(self.d_volume_module, axis=2)
                            projection = cp.asnumpy(projection)
                        except Exception as cuda_error:
                            print(f"CUDA error in YZ_MAX_PROJECTION, falling back to CPU: {cuda_error}")
                            # Fallback to CPU computation
                            volume_cpu = cp.asnumpy(self.d_volume_module)
                            projection = np.max(volume_cpu, axis=2)
                        
                        stat_plane(projection, label="YZ_MAX_PROJECTION")
                        image = Image.fromarray(_to_uint8(projection))
                        return self._apply_additional_display(image, additional_display, display_type, plane_number, 'YZ')
                    except Exception as e:
                        # print(f"Error in YZ_MAX_PROJECTION: {e}")
                        pass
                return _fallback("YZ_MAX_PROJECTION")

            # Default: show raw hologram
            arr = _get_raw_array()
            return Image.fromarray(_to_uint8(arr))

        except Exception as e:
            # Minimal error reporting
            # print(f"Error in get_display_image: {e}")
            try:
                arr = self.h_raw_holo if self.h_raw_holo is not None else np.zeros((100,100), dtype=np.float64)
                return Image.fromarray(_to_uint8(arr))
            except:
                return Image.fromarray(np.zeros((100, 100), dtype=np.uint8))

    def set_parameter(self, name, value):
        setattr(self, name, value)
        return f"{name} updated"
    
    def get_parameter(self, name, default=None):
        """Get a parameter value"""
        return getattr(self, name, default)

    def open_hologram_image(self, directory, filename):
        filepath = os.path.join(directory, filename)
        img = Image.open(filepath)
        # Resize if parameters are present
        x = int(self.holo_size_x)
        y = int(self.holo_size_y)
        img = img.resize((x, y))
        return img

    def compute_mean_hologram(self, directory, image_type, progress_callback=None):
        """Compute mean hologram from all images in directory"""
        ext_map = {"BMP": ".bmp", "TIF": ".tif", "JPG": ".jpg", "PNG": ".png"}
        ext = ext_map[image_type]
        
        if not os.path.isdir(directory):
            raise ValueError("Invalid directory")
            
        image_files = [f for f in os.listdir(directory) if f.lower().endswith(ext.lower())]
        
        if not image_files:
            raise ValueError("No images found")
            
        total_images = len(image_files)
        mean_image = None
        
        for i, filename in enumerate(image_files):
            filepath = os.path.join(directory, filename)
            try:
                img = Image.open(filepath).convert('L')  # Convert to grayscale
                img_array = np.array(img, dtype=np.float64)
                
                if mean_image is None:
                    mean_image = img_array
                else:
                    mean_image += img_array
                    
                # Update progress
                if progress_callback:
                    progress_callback(i+1, total_images)
                    
            except Exception as e:
                raise Exception(f"Error processing {filename}: {str(e)}")
        
        # Finalize mean (keep in [0,255] range)
        mean_image = mean_image / total_images
        
        # Create mean directory
        mean_dir = os.path.join(directory, "mean")
        os.makedirs(mean_dir, exist_ok=True)
        
        # Save as .tif for calculations (float32 precision, [0,255] range)
        mean_tif_path = os.path.join(mean_dir, "mean_hologram.tif")
        mean_image_float32 = mean_image.astype(np.float32)
        mean_pil = Image.fromarray(mean_image_float32, mode='F')  # 'F' mode for 32-bit float
        mean_pil.save(mean_tif_path)
        
        # Save as .bmp for visualization (uint8, [0,255] range)
        mean_bmp_path = os.path.join(mean_dir, "mean_hologram.bmp")
        mean_image_uint8 = np.clip(mean_image, 0, 255).astype(np.uint8)
        mean_pil_vis = Image.fromarray(mean_image_uint8, mode='L')
        mean_pil_vis.save(mean_bmp_path)
        
        return mean_tif_path  # Return the .tif path for use in calculations

    def batch_process(self):
        if not self.holograms_directory:
            raise ValueError("Holograms directory not specified")
        # Simulate processing
        self.results = {
            "localizations": [(0.1, 0.2, 0.3), (0.4, 0.5, 0.6), (0.7, 0.8, 0.9)],
            "images": ["image1.png", "image2.png"]
        }
        return "Processing completed"

    def cancel_batch(self):
        self.results = {}
        return "Processing cancelled"
    
    def enter_test_mode(self):
        """Entre en mode test"""
        self.mode = 'TEST'
        return "Test mode activated"

    def exit_test_mode(self):
        """Sort du mode test"""
        self.mode = 'IDLE'
        self.cleanup_test_mode()
        return "Test mode deactivated"
    
    def enter_batch_mode(self):
        """Entre en mode batch"""
        self.mode = 'BATCH'
        self.batch_first_hologram_done = False
        return "Batch mode activated"
    
    def exit_batch_mode(self):
        """Sort du mode batch"""
        self.mode = 'IDLE'
        self.batch_first_hologram_done = False
        return "Batch mode deactivated"
    
    def allocate(self):
        """Allocation mémoire pour le traitement des hologrammes"""
        import cupy as cp
        
        # Basic parameters
        cam_nb_pix_X = int(self.holo_size_x)
        cam_nb_pix_Y = int(self.holo_size_y)
        nb_plane = int(self.number_of_planes)
        
        # Array allocation according to new architecture
        self.h_raw_holo = np.zeros(shape=(cam_nb_pix_Y, cam_nb_pix_X), dtype=np.float32)
        self.h_mean_holo = None
        self.h_cleaned_holo = None
        self.h_filtered_holo = np.zeros(shape=(cam_nb_pix_Y, cam_nb_pix_X), dtype=np.float32)
        self.d_holo = cp.zeros(shape=(cam_nb_pix_Y, cam_nb_pix_X), dtype=cp.complex64)
        self.d_filtered_holo = cp.zeros(shape=(cam_nb_pix_Y, cam_nb_pix_X), dtype=cp.float32)
        self.d_fft_holo = cp.zeros(shape=(cam_nb_pix_Y, cam_nb_pix_X), dtype=cp.complex64)
        self.d_fft_holo_filtered = cp.zeros(shape=(cam_nb_pix_Y, cam_nb_pix_X), dtype=cp.complex64)
        self.d_fft_holo_propag = cp.zeros(shape=(cam_nb_pix_Y, cam_nb_pix_X), dtype=cp.complex64)
        self.d_holo_propag = cp.zeros(shape=(cam_nb_pix_Y, cam_nb_pix_X), dtype=cp.complex64)
        self.d_KERNEL = cp.zeros(shape=(cam_nb_pix_Y, cam_nb_pix_X), dtype=cp.complex64)
        self.d_FFT_KERNEL = cp.zeros(shape=(cam_nb_pix_Y, cam_nb_pix_X), dtype=cp.complex64)
        self.d_volume_module = cp.zeros(shape=(nb_plane, cam_nb_pix_Y, cam_nb_pix_X), dtype=cp.float32)
        self.d_bin_volume_focus = cp.zeros(shape=(nb_plane, cam_nb_pix_Y, cam_nb_pix_X), dtype=cp.dtype(bool))
        
        # Charger l'hologramme moyen en GPU si disponible
        if self.h_mean_holo is not None:
            self.d_mean_holo = cp.asarray(self.h_mean_holo)
        else:
            self.d_mean_holo = None
    
        self.memory_allocated = True
        
        # Print parameters for debugging
        self.print_parameters()

    def check_reallocation_needed(self, new_params):
        """Check if GPU memory reallocation is needed based on parameter changes"""
        if not self.memory_allocated:
            return True
            
        # Parameters that require reallocation
        size_changing_params = ["holo_size_x", "holo_size_y", "number_of_planes"]
        
        for param in size_changing_params:
            if param in new_params:
                old_value = getattr(self, param)
                new_value = new_params[param]
                if old_value != new_value:
                    return True
        return False

    def update_parameters_and_reallocate_if_needed(self, new_params):
        """Update parameters and reallocate GPU memory if needed"""
        if self.check_reallocation_needed(new_params):
            # Cleanup existing allocation
            if self.memory_allocated:
                self.cleanup_test_mode()
            
            # Update parameters
            self.set_parameters(**new_params)
            
            # Reallocate with new parameters
            result = self.allocate()
            return result
        else:
            # Just update parameters, no reallocation needed
            self.set_parameters(**new_params)
            return "Parameters updated without reallocation"

    def process_hologram_complete_pipeline(self, directory, filename):
        """
        Complete hologram processing pipeline following test_HoloTracker_locate.py
        Pipeline: Load -> Remove Mean -> Volume Propagation -> Focus -> CCL3D -> Label Analysis
        """
        # print(f"🚀 Core: Starting hologram processing for {filename}")
        
        if not CUPY_AVAILABLE:
            # print("❌ Core: CuPy not available")
            # Populate results with an explicit error so UI can display a meaningful message
            self.results = {
                'number_of_objects': 0,
                'features': np.array([]),
                'processing_times': {'total_processing': 0.0},
                'error': 'CuPy not available for GPU processing'
            }
            return "Error: CuPy not available for GPU processing"
            
        if not self.memory_allocated:
            # print("❌ Core: Test mode not initialized")
            # Ensure results carry an explicit error to avoid empty dicts
            self.results = {
                'number_of_objects': 0,
                'features': np.array([]),
                'processing_times': {'total_processing': 0.0},
                'error': 'Test mode not initialized (allocate must be called before processing)'
            }
            return "Error: Test mode not initialized"
            
        # Print parameters for debugging
        self.print_parameters()
            
        try:
            import time
            
            # 1. Load raw hologram and preprocessing
            start_processing = time.perf_counter()
            filepath = os.path.join(directory, filename)
            cam_nb_pix_X = int(self.holo_size_x)
            cam_nb_pix_Y = int(self.holo_size_y)
            
            # Load raw hologram into h_raw_holo
            self.h_raw_holo[:] = read_image(filepath, cam_nb_pix_X, cam_nb_pix_Y)
            stat_plane(self.h_raw_holo, label="h_raw_holo after loading")
            
            # 2. Clean hologram (remove mean according to parameter and type)
            remove_mean = self.remove_mean
            if remove_mean:
                cleaning_type = self.cleaning_type
                # Load mean hologram if needed
                if self.h_mean_holo is None:
                    self.load_mean_hologram()
                
                # Debug info about mean hologram
                if self.h_mean_holo is not None:
                    mean_min, mean_max = self.h_mean_holo.min(), self.h_mean_holo.max()
                    print(f"🧹 DEBUG: h_mean_holo available: shape={self.h_mean_holo.shape}, mean_range=[{mean_min:.3f}, {mean_max:.3f}]")
                else:
                    print(f"🧹 DEBUG: h_mean_holo is None!")
                
                print(f"🧹 DEBUG: remove_mean parameter = {remove_mean}")
                print(f"🧹 DEBUG: cleaning_type parameter = {cleaning_type}")
                
                # Apply cleaning based on type
                if cleaning_type == "subtraction":
                    input_min, input_max = self.h_raw_holo.min(), self.h_raw_holo.max()
                    mean_min, mean_max = self.h_mean_holo.min(), self.h_mean_holo.max()
                    
                    self.h_cleaned_holo = self.h_raw_holo - self.h_mean_holo
                    cleaned_min_before = self.h_cleaned_holo.min()
                    self.h_cleaned_holo = self.h_cleaned_holo - self.h_cleaned_holo.min()  # Ensure non-negative
                    cleaned_min_after, cleaned_max_after = self.h_cleaned_holo.min(), self.h_cleaned_holo.max()
                    
                    print(f"🧹 Core: Input range: [{input_min:.1f}, {input_max:.1f}]")
                    print(f"🧹 Core: Mean range: [{mean_min:.1f}, {mean_max:.1f}]")
                    print(f"🧹 Core: Cleaned range: [{cleaned_min_after:.1f}, {cleaned_max_after:.1f}]")
                    
                else:  # division (default)
                    input_min, input_max = self.h_raw_holo.min(), self.h_raw_holo.max()
                    mean_min, mean_max = self.h_mean_holo.min(), self.h_mean_holo.max()
                    
                    self.h_cleaned_holo = self.h_raw_holo.astype(np.float64) / (self.h_mean_holo + 1e-10)
                    self.h_cleaned_holo = np.power(self.h_cleaned_holo, 0.8).astype(np.float32)  # Limit extreme values
                    cleaned_min, cleaned_max = self.h_cleaned_holo.min(), self.h_cleaned_holo.max()
                    
                    print(f"🧹 Core: Input range: [{input_min:.1f}, {input_max:.1f}]")
                    print(f"🧹 Core: Mean range: [{mean_min:.1f}, {mean_max:.1f}]") 
                    print(f"🧹 Core: Cleaned range (division): [{cleaned_min:.3f}, {cleaned_max:.3f}]")
            else:
                print(f"🧹 DEBUG: remove_mean = False, using raw hologram")
                self.h_cleaned_holo[:] = self.h_raw_holo
            
            # Transfer cleaned hologram to GPU - reuse pre-allocated array
            self.d_holo[:] = cp.asarray(self.h_cleaned_holo.astype(cp.complex64))
            
            t1 = time.perf_counter()
            t_preprocess = t1 - start_processing
            
            # 3. Spectral filtering before propagation
            
            # Get filtering parameters from UI
            f_pix_min = int(self.high_pass)  # high_pass = f_pix_min
            f_pix_max = int(self.low_pass)  # low_pass = f_pix_max
                    
            # 4. Volume propagation by angular spectrum method (without filtering since already done)
            
            # Get parameters for propagation
            wavelength = float(self.wavelength)
            medium_optical_index = float(self.medium_optical_index)
            medium_wavelength = wavelength / medium_optical_index
            objective_magnification = float(self.objective_magnification)
            pixel_size = float(self.pixel_size)
            nb_plane = int(self.number_of_planes)
            dx = pixel_size / objective_magnification # in meters
            dy = pixel_size / objective_magnification # in meters
            dz = float(self.step)  # in meters
            distance_ini = float(self.distance_ini)

            propag_start = time.perf_counter()

            stat_plane(self.d_holo, "d_HOLO  before propagation")

            propag.volume_propag_angular_spectrum_to_module(
                self.d_holo, self.d_fft_holo, self.d_fft_holo_filtered,self.d_KERNEL, 
                self.d_filtered_holo, self.d_fft_holo_propag, self.d_holo_propag, 
                self.d_volume_module, medium_wavelength, objective_magnification, 
                pixel_size, cam_nb_pix_X, cam_nb_pix_Y, distance_ini, dz, nb_plane, f_pix_min, f_pix_max)
            
            stat_plane(self.d_holo, "d_HOLO  after propagation")
            stat_plane(self.d_volume_module, "d_volume before focus")
            
            # Copy filtered hologram to CPU for display purposes
            if self.d_filtered_holo is not None:
                self.h_filtered_holo[:] = cp.asnumpy(self.d_filtered_holo)
                   
            t2 = time.perf_counter()
            t_propag = t2 - propag_start
            
            # 4. Focus on the volume (INPLACE)
            focus_start = time.perf_counter()
            sum_size = int(self.sum_size)
            
            # Get focus type from parameters
            focus_type_str = self.focus_type
            focus_type_map = {
                "SUM_OF_INTENSITY": Focus_type.SUM_OF_INTENSITY,
                "SUM_OF_LAPLACIAN": Focus_type.SUM_OF_LAPLACIAN,
                "SUM_OF_VARIANCE": Focus_type.SUM_OF_VARIANCE,
                "TENEGRAD": Focus_type.TENEGRAD
            }
            focus_type_enum = focus_type_map[focus_type_str]
            
            # print(f"🎯 Core: Applying focus type: {focus_type_str} (enum: {focus_type_enum})")
            focus.focus(self.d_volume_module, self.d_volume_module, sum_size, focus_type_enum)
            
            stat_plane(self.d_volume_module, "d_volume after focus")

            t3 = time.perf_counter()
            t_focus = t3 - focus_start
            
            # 5. Compute threshold and CCL3D
            ccl_start = time.perf_counter()
            nb_StdVar_threshold = float(self.nb_StdVar_threshold)
            n_connectivity = int(self.connectivity)
            
            # NEW LOGIC: Threshold recalculation based on mode
            # Base conditions (always needed)
            base_need = (not self.threshold or 
                        not self.last_nb_StdVar_threshold or
                        self.last_nb_StdVar_threshold != nb_StdVar_threshold)
            
            # Mode-specific logic
            if self.mode == 'TEST':
                # TEST mode: always recalculate threshold
                need_recalc = True
            elif self.mode == 'BATCH':
                # BATCH mode: depends on batch_threshold setting
                batch_recalc = self.batch_threshold == "compute on each hologram"
                if self.batch_threshold == "compute on 1st hologram":
                    # Only on first hologram in batch
                    need_recalc = base_need or not self.batch_first_hologram_done
                else:
                    # On each hologram (batch_recalc = True)
                    need_recalc = base_need or batch_recalc
            else:
                # IDLE or other modes: use base logic
                need_recalc = base_need
            
            if need_recalc:
                print(f"🎯 DEBUG: Recalculating threshold...")
                print(f"🎯 DEBUG: Reason - mode: {self.mode}, threshold: {self.threshold}, last_nb_StdVar: {self.last_nb_StdVar_threshold}")
                self.threshold = calc_threshold(self.d_volume_module, nb_StdVar_threshold)
                self.last_nb_StdVar_threshold = nb_StdVar_threshold
                # If we computed for the first hologram in batch, mark it done
                if self.mode == 'BATCH' and self.batch_threshold == "compute on 1st hologram":
                    self.batch_first_hologram_done = True
            else:
                print(f"🎯 DEBUG: Using cached threshold: {self.threshold}")
                
            print(f"🔍 DEBUG Threshold: {self.threshold} (nb_StdVar_threshold={nb_StdVar_threshold})")
            
            # CCL3D
            d_labels, number_of_labels = CCL3D(
                self.d_bin_volume_focus, self.d_volume_module, 
                type_threshold.THRESHOLD, self.threshold, n_connectivity
            )
            
            print(f"🔍 DEBUG CCL3D result: {number_of_labels} labels found")
            
            t4 = time.perf_counter()
            t_ccl = t4 - ccl_start
            
            # 6. Label analysis (centroid computation)
            cca_start = time.perf_counter()
            if number_of_labels > 0:
                features = np.ndarray(shape=(number_of_labels,), dtype=dobjet)
                
                print(f"🔍 DEBUG Before CCA: {number_of_labels} labels, dx={dx}, dy={dy}, dz={dz}")
                
                features = CCA_CUDA_float(
                    d_labels, self.d_volume_module, number_of_labels, 
                    1, cam_nb_pix_X, cam_nb_pix_Y, nb_plane, dx, dy, dz
                )
                
                print(f"🔍 DEBUG After CCA: {len(features) if features is not None else 0} features")
                
                # Filtrage par nombre de voxels
                min_voxel = int(self.min_voxel)
                max_voxel = int(self.max_voxel)
                print(f"🔍 DEBUG Filter params: min_voxel={min_voxel}, max_voxel={max_voxel}")
                
                if min_voxel != 0 or max_voxel != 0:
                    features_before_filter = len(features) if features is not None else 0
                    features = CCL_filter(features, min_voxel, max_voxel)
                    features_after_filter = len(features) if features is not None else 0
                    print(f"🔍 DEBUG Filter result: {features_before_filter} -> {features_after_filter} objects")

            else:
                features = np.array([])
                print(f"🔍 DEBUG No labels found, features set to empty array")
                
            t5 = time.perf_counter()
            t_cca = t5 - cca_start
            t_total = t5 - start_processing
            
            # TIME DISPLAY LIKE test_HoloTracker_locate.py
            # print(f'number of objects located: {number_of_labels}')
            # print(f't preprocess: {t_preprocess:.6f}')
            # print(f't propagation: {t_propag:.6f}')
            # print(f't focus: {t_focus:.6f}') 
            # print(f't ccl: {t_ccl:.6f}')
            # print(f't cca: {t_cca:.6f}')
            # print(f'total iteration time: {t_total:.6f}')
            # print(f'---')
            
            # Store results
            self.results = {
                'number_of_objects': number_of_labels,
                'features': features,
                'processing_times': {
                    'preprocessing': t_preprocess,
                    'propagation': t_propag,
                    'focus': t_focus,
                    'ccl': t_ccl,
                    'cca': t_cca,
                    'total_processing': t_total
                }
            }
            
            # print(f"✅ Core: Results stored with processing_times: {self.results['processing_times']}")
            
            if number_of_labels > 0:
                return f"Processing completed: {number_of_labels} objects found"
            else:
                return "Processing completed: No objects found"
                
        except Exception as e:
            print(e)      
            return f"Error in processing pipeline: {e}"

    def get_3d_results_data(self):
        """Get 3D results data for display in UI tab (no pop-up)"""
        try:
            # Initialize result data structure
            result_data = {
                'localizations': [],
                'particle_sizes': [],
                'count': 0
            }
            
            # Debug: Check what self.results contains
            try:
                keys = list(self.results.keys()) if self.results and isinstance(self.results, dict) else []
            except Exception:
                keys = []
            print(f"🔧 Core: self.results keys: {keys}")

            # Add processing times only if available to avoid KeyError
            if self.results and isinstance(self.results, dict) and 'processing_times' in self.results:
                result_data['processing_times'] = self.results['processing_times']
            
            # Add detection results if available
            if self.results and 'features' in self.results and self.results['features'] is not None:
                features = self.results['features']
                if len(features) > 0:
                    # Extract coordinates from features (following test_HoloTracker_locate.py format)
                    positions = pd.DataFrame(features, columns=['i_image', 'baryX', 'baryY', 'baryZ', 'nb_pix'])
                    
                    # Return localizations in the format expected by UI (convert from meters to micrometers)
                    localizations = []
                    for _, row in positions.iterrows():
                        # Convert from meters to micrometers for UI display
                        x_um = row['baryX'] * 1e6
                        y_um = row['baryY'] * 1e6  
                        z_um = row['baryZ'] * 1e6
                        localizations.append((x_um, y_um, z_um))
                    
                    result_data.update({
                        'localizations': localizations,
                        'particle_sizes': positions['nb_pix'].tolist(),
                        'count': len(features)
                    })
                
            return result_data
            
        except Exception as e:
            print(f"Error getting 3D results data: {e}")
            # Return basic structure with timing if available (guard self.results)
            basic_data = {'localizations': [], 'particle_sizes': [], 'count': 0}
            if self.results and isinstance(self.results, dict) and 'processing_times' in self.results:
                basic_data['processing_times'] = self.results['processing_times']
            return basic_data

    def show_3d_results(self):
        """Legacy method - now returns data for UI display instead of pop-up"""
        data = self.get_3d_results_data()
        if data is None:
            return "No results to display"
        return f"3D data ready with {data['count']} particles"

    def get_cleaned_hologram_image(self, directory, filename):
        """Get cleaned hologram image for display"""
        try:
            filepath = os.path.join(directory, filename)
            cam_nb_pix_X = int(self.holo_size_x)
            cam_nb_pix_Y = int(self.holo_size_y)
            
            # Load hologram
            h_holo = read_image(filepath, cam_nb_pix_X, cam_nb_pix_Y)
            
            # Apply mean removal if available
            if self.h_mean_holo is not None:
                h_holo_cleaned = h_holo / self.h_mean_holo
                
                # Normalize for display
                min_val = h_holo_cleaned.min()
                max_val = h_holo_cleaned.max()
                h_holo_normalized = ((h_holo_cleaned - min_val) * 255 / (max_val - min_val)).astype(np.uint8)
                
                return Image.fromarray(h_holo_normalized)
            else:
                # Return original if no mean hologram
                min_val = h_holo.min()
                max_val = h_holo.max()
                h_holo_normalized = ((h_holo - min_val) * 255 / (max_val - min_val)).astype(np.uint8)
                return Image.fromarray(h_holo_normalized)
                
        except Exception as e:
            # print(f"Error getting cleaned hologram: {e}")
            return None

    def add_detection_markers_to_image(self, image, detections):
        """Add red markers to show detected objects on hologram"""
        try:
            import cv2
            
            # Convert PIL image to OpenCV format
            if isinstance(image, Image.Image):
                img_array = np.array(image)
            else:
                img_array = image
                
            # Convert to RGB if grayscale
            if len(img_array.shape) == 2:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            
            print(f"🔍 DEBUG add_detection_markers called - image shape: {img_array.shape}")
            
            # Add red circles for each detection
            if 'features' in self.results and self.results['features'] is not None:
                features = self.results['features']
                print(f"🔍 DEBUG Found {len(features)} features to display")
            else:
                print(f"🔍 DEBUG No features found in results")
                print(f"  self.results keys: {list(self.results.keys()) if self.results else 'None'}")
            
            # Add red circles for each detection
            if 'features' in self.results and self.results['features'] is not None:
                features = self.results['features']
                print(f"🔍 DEBUG Found {len(features)} features to display")
                
                for i, feature in enumerate(features):
                    try:
                        if i < 3:  # Only show detailed debug for first 3 features
                            print(f"🔍 DEBUG Processing feature {i+1}: {feature}")
                        
                        # Extract coordinates (baryX, baryY are in meters, convert to pixels)
                        x_m = feature[1]  # baryX in meters
                        y_m = feature[2]  # baryY in meters
                        
                        if i < 3:
                            print(f"🔍 DEBUG Extracted coordinates: x_m={x_m}, y_m={y_m}")
                        
                        # Convert using the same dx, dy resolution as used in processing
                        dx = float(self.pixel_size) / float(self.objective_magnification)  # meters per pixel
                        dy = float(self.pixel_size) / float(self.objective_magnification)  # meters per pixel
                        
                        if i < 3:
                            print(f"🔍 DEBUG Spatial resolution: dx={dx} m/pixel, dy={dy} m/pixel")
                        
                        # Convert from physical coordinates (meters) to pixel coordinates
                        x_pix = int(x_m / dx)
                        y_pix = int(y_m / dy)
                        
                        if i < 3:
                            print(f"🔍 DEBUG Pixel coordinates: x_pix={x_pix}, y_pix={y_pix}")
                        
                        # Draw red circle only if coordinates are within image bounds
                        if 0 <= x_pix < img_array.shape[1] and 0 <= y_pix < img_array.shape[0]:
                            cv2.circle(img_array, (x_pix, y_pix), 5, (255, 0, 0), 2)
                            if i < 3:
                                print(f"  ✅ Circle drawn at ({x_pix}, {y_pix})")
                        else:
                            if i < 3:
                                print(f"  ❌ Circle NOT drawn - coordinates out of bounds: ({x_pix}, {y_pix}) vs image size {img_array.shape}")
                        
                        # Show summary message after first 3 features
                        if i == 2 and len(features) > 3:
                            print(f"🔍 DEBUG ... (continuing to draw remaining {len(features)-3} features without detailed debug)")
                            
                    except Exception as e:
                        if i < 3:
                            print(f"❌ ERROR processing feature {i+1}: {e}")
                            print(f"   Feature data: {feature}")
                            import traceback
                            traceback.print_exc()
            
            return Image.fromarray(img_array)
            
        except ImportError:
            # print("OpenCV not available, cannot add detection markers")
            return image
        except Exception as e:
            # print(f"Error adding detection markers: {e}")
            return image

    def add_detection_markers_to_image_2d_projection(self, image, results, projection_type):
        """Ajoute des marqueurs de détection sur une projection 2D selon le type de projection"""
        try:
            import cv2
            
            # Convert PIL image to OpenCV format
            if isinstance(image, Image.Image):
                img_array = np.array(image)
            else:
                img_array = image
                
            # Convert to RGB if grayscale
            if len(img_array.shape) == 2:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            
            # Add markers for each detected particle
            if 'features' in self.results and self.results['features'] is not None:
                features = self.results['features']
                for feature in features:
                    # Extract coordinates (now in meters, convert to micrometers)
                    x_m = feature[1]  # baryX in meters
                    y_m = feature[2]  # baryY in meters
                    z_m = feature[3]  # baryZ in meters
                    
                    # Convert from meters to micrometers
                    x_um = x_m * 1e6
                    y_um = y_m * 1e6
                    z_um = z_m * 1e6
                    
                    # Convert using the same dx, dy resolution as used in processing
                    dx = float(self.pixel_size) / float(self.objective_magnification)  # meters per pixel
                    dy = float(self.pixel_size) / float(self.objective_magnification)  # meters per pixel
                    dz = float(self.step)  # Step size in meters
                    
                    # X and Y coordinates in pixels (lateral dimensions)
                    x_pix = int(x_m / dx)
                    y_pix = int(y_m / dy)
                    
                    # Z coordinate in pixels (depth dimension) - convert from meters to plane index
                    z_pix = int(z_m / dz)  # Direct conversion from meters to plane index
                    
                    # Choose coordinates based on projection type
                    if projection_type == 'XY':
                        coord_x, coord_y = x_pix, y_pix
                    elif projection_type == 'XZ':
                        coord_x, coord_y = x_pix, z_pix
                    elif projection_type == 'YZ':
                        coord_x, coord_y = y_pix, z_pix
                    else:
                        coord_x, coord_y = x_pix, y_pix
                    
                    # Clamp coordinates to image boundaries to prevent IndexError
                    coord_x = max(0, min(coord_x, img_array.shape[1] - 1))
                    coord_y = max(0, min(coord_y, img_array.shape[0] - 1))
                    
                    # Draw markers (green circle + red center)
                    cv2.circle(img_array, (coord_x, coord_y), 5, (0, 255, 0), 2)
                    cv2.circle(img_array, (coord_x, coord_y), 2, (255, 0, 0), -1)
            
            return Image.fromarray(img_array)
            
        except ImportError:
            # print("OpenCV not available, cannot add detection markers")
            return image
        except Exception as e:
            # print(f"Error adding detection markers to projection: {e}")
            return image

    def _apply_additional_display(self, image, additional_display, display_type, plane_number, projection_type=None):
        """Apply additional display overlay based on the selected option"""
        try:
            # No overlays for FFT displays
            if display_type in ["FFT_HOLOGRAM", "FFT_FILTERED_HOLOGRAM"]:
                return image
                
            if additional_display == "None":
                return image
            elif additional_display == "Centroid positions":
                return self._add_centroid_overlay(image, display_type, plane_number, projection_type)
            elif additional_display == "Segmentation":
                return self._add_segmentation_overlay(image, display_type, plane_number, projection_type)
            else:
                return image
        except Exception as e:
            # print(f"Error applying additional display: {e}")
            return image

    def _add_centroid_overlay(self, image, display_type, plane_number, projection_type=None):
        """Add centroid position markers to the image"""
        try:
            if display_type == "VOLUME_PLANE_NUMBER":
                # For volume plane: show only markers for the specific plane
                return self._add_centroid_markers_for_plane(image, plane_number)
            elif projection_type:
                # For projections: use the appropriate projection method
                return self.add_detection_markers_to_image_2d_projection(image, self.results, projection_type)
            else:
                # For hologram images: use standard marker method
                return self.add_detection_markers_to_image(image, self.results)
        except Exception as e:
            # print(f"Error adding centroid overlay: {e}")
            return image

    def _add_segmentation_overlay(self, image, display_type, plane_number, projection_type=None):
        """Add segmentation overlay to the image"""
        try:
            if self.d_bin_volume_focus is None:
                return image

            # Helper: normalize a numpy array to uint8 for display
            def _to_uint8(arr):
                arr = np.array(arr)  # ensure numpy
                if np.iscomplexobj(arr):
                    arr = np.abs(arr)
                arr = arr.astype(np.float64)
                mx = arr.max() if arr.size else 0.0
                mn = arr.min() if arr.size else 0.0
                if mx > mn:
                    out = ((arr - mn) * 255.0 / (mx - mn)).astype(np.uint8)
                elif mx > 0:
                    out = (arr * 255.0 / mx).astype(np.uint8)
                else:
                    out = np.zeros_like(arr, dtype=np.uint8)
                return out

            if display_type == "VOLUME_PLANE_NUMBER":
                # For volume plane: show segmentation for the specific plane
                if plane_number < self.d_bin_volume_focus.shape[0]:
                    bin_plane = cp.asnumpy(self.d_bin_volume_focus[plane_number, :, :])
                    return self._blend_segmentation_with_image(image, bin_plane)
            elif projection_type == 'XY':
                # Sum or max projection along Z axis (axis=0) using projection_bool
                bin_projection = projection_bool(self.d_bin_volume_focus, axis=0)
                bin_projection = cp.asnumpy(bin_projection)
                return self._blend_segmentation_with_image(image, bin_projection)
            elif projection_type == 'XZ':
                # Sum or max projection along Y axis (axis=1) using projection_bool
                bin_projection = projection_bool(self.d_bin_volume_focus, axis=1)
                bin_projection = cp.asnumpy(bin_projection)
                return self._blend_segmentation_with_image(image, bin_projection)
            elif projection_type == 'YZ':
                # Sum or max projection along X axis (axis=2) using projection_bool
                bin_projection = projection_bool(self.d_bin_volume_focus, axis=2)
                bin_projection = cp.asnumpy(bin_projection)
                return self._blend_segmentation_with_image(image, bin_projection)
            else:
                # For hologram images: use XY projection with projection_bool
                bin_projection = projection_bool(self.d_bin_volume_focus, axis=0)
                bin_projection = cp.asnumpy(bin_projection)
                return self._blend_segmentation_with_image(image, bin_projection)

            return image
        except Exception as e:
            # print(f"Error adding segmentation overlay: {e}")
            return image

    def _add_centroid_markers_for_plane(self, image, plane_number):
        """Add centroid markers for a specific plane only"""
        try:
            import cv2
            
            # Convert PIL image to OpenCV format
            if isinstance(image, Image.Image):
                img_array = np.array(image)
            else:
                img_array = image
                
            # Convert to RGB if grayscale
            if len(img_array.shape) == 2:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            
            # Add red circles for each detection on this specific plane
            if 'features' in self.results and self.results['features'] is not None:
                features = self.results['features']
                for feature in features:
                    # Extract coordinates (baryX, baryY, baryZ are now in meters)
                    x_m = feature[1]  # baryX in meters
                    y_m = feature[2]  # baryY in meters
                    z_m = feature[3]  # baryZ in meters
                    
                    # Convert from meters to micrometers
                    x_um = x_m * 1e6
                    y_um = y_m * 1e6
                    z_um = z_m * 1e6
                    
                    # Convert Z position to plane number
                    cam_pix_size = float(self.pixel_size)  # Use same parameter name
                    cam_magnification = float(self.objective_magnification)  # Use same parameter name
                    dz = float(self.step)  # Step size in meters
                    
                    # Calculate the plane number for this feature
                    feature_plane = int(z_um / (dz * 1e6))  # Convert to plane index
                    
                    # Only show markers for features in the current plane (with some tolerance)
                    if abs(feature_plane - plane_number) <= 1:  # Allow ±1 plane tolerance
                        # Convert from micrometers to pixels
                        effective_pixel_size = cam_pix_size / cam_magnification
                        x_pix = int(x_um * 1e-6 / effective_pixel_size)
                        y_pix = int(y_um * 1e-6 / effective_pixel_size)
                        
                        # Draw red circle
                        cv2.circle(img_array, (x_pix, y_pix), 10, (255, 0, 0), 2)
            
            return Image.fromarray(img_array)
        except Exception as e:
            # print(f"Error adding markers for plane: {e}")
            return image

    def _blend_segmentation_with_image(self, image, segmentation_data):
        """Blend segmentation data as overlay with the original image"""
        try:
            import cv2
            
            # Convert PIL image to OpenCV format
            if isinstance(image, Image.Image):
                img_array = np.array(image)
            else:
                img_array = image
                
            # Convert to RGB if grayscale
            if len(img_array.shape) == 2:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            
            # Debug information
            print(f"Debug blend: img_array shape: {img_array.shape}, segmentation shape: {segmentation_data.shape}")
            
            # Ensure segmentation_data matches image dimensions
            img_h, img_w = img_array.shape[:2]
            seg_h, seg_w = segmentation_data.shape
            
            if seg_h != img_h or seg_w != img_w:
                # Resize segmentation to match image
                segmentation_data = cv2.resize(segmentation_data.astype(np.float32), (img_w, img_h))
                print(f"Debug blend: Resized segmentation to {segmentation_data.shape}")
            
            # Create binary mask for segmentation
            if segmentation_data.max() > 0:
                # Create binary mask where segmentation is present
                mask = (segmentation_data > 0).astype(np.uint8)
                print(f"Debug blend: Found {np.sum(mask)} non-zero pixels in mask")
            else:
                mask = np.zeros_like(segmentation_data, dtype=np.uint8)
                print("Debug blend: No segmentation data found")
            
            # Create result image starting with original
            result = img_array.copy()
            
            # Apply solid blue color where segmentation is present
            result[mask > 0, 2] = 255  # Blue channel at maximum
            result[mask > 0, 0] = 0    # Red channel to 0
            result[mask > 0, 1] = 0    # Green channel to 0
            
            return Image.fromarray(result)
        except Exception as e:
            # print(f"Error blending segmentation: {e}")
            return image

    def get_default_display_type(self):
        """Get default display type based on parameters"""
        remove_mean = self.remove_mean
        # Check if remove_mean is enabled (regardless of mean_hologram availability)
        if remove_mean:
            return "CLEANED_HOLOGRAM"
        else:
            return "RAW_HOLOGRAM"
        
    def cleanup_test_mode(self):
        """Nettoie la mémoire allouée pour le mode test"""
        # Free GPU and CPU memory
        gpu_vars = ['d_holo', 'd_filtered_holo', 'd_fft_holo', 'd_fft_holo_filtered', 'd_fft_holo_propag', 'd_holo_propag', 
                   'd_KERNEL', 'd_FFT_KERNEL', 'd_volume_module', 'd_bin_volume_focus', 'd_mean_holo']
        
        cpu_vars = ['h_raw_holo', 'h_cleaned_holo', 'h_mean_holo', 'h_filtered_holo']
        
        # Nettoyer les variables GPU
        for var_name in gpu_vars:
            var = getattr(self, var_name)
            if var is not None:
                del var
            setattr(self, var_name, None)
        
        # Nettoyer les variables CPU
        for var_name in cpu_vars:
            var = getattr(self, var_name)
            if var is not None:
                del var
            setattr(self, var_name, None)
        
        self.memory_allocated = False
        
    def _initialize_propagation_kernels(self):
        """Initialize propagation kernels using existing functions from test_HoloTracker_locate.py"""
        try:
            # The propagation kernel is calculated dynamically in volume_propag_angular_spectrum_to_module
            # We only need to allocate memory space as in test_HoloTracker_locate.py
            # print("Status: Propagation kernels space allocated successfully")
            # The d_KERNEL is already allocated in allocate, no need to recalculate it here
            pass
            
        except Exception as e:
            # print(f"Warning: Could not initialize propagation kernels: {e}")
            # Fallback - use existing kernel or create if needed
            cam_nb_pix_X = int(self.holo_size_x)
            cam_nb_pix_Y = int(self.holo_size_y)
            if self.d_KERNEL.shape != (cam_nb_pix_Y, cam_nb_pix_X):
                self.d_KERNEL = cp.zeros(shape=(cam_nb_pix_Y, cam_nb_pix_X), dtype=cp.complex64)

    def extract_object_slices(self, pos_x_um, pos_y_um, pos_z_um, vox_xy, vox_z):
        """Extract 3 slice views (XY, XZ, YZ) around an object from the reconstructed volume"""
        if not self.memory_allocated or self.d_volume_module is None:
            raise ValueError("Test mode not initialized or no volume available")
            
        # print(f"🔍 Core: Extracting slices at ({pos_x_um:.3f}, {pos_y_um:.3f}, {pos_z_um:.3f}) µm")
        
        # Convert micrometers to pixel coordinates using the same formula as in add_detection_markers
        volume_shape = self.d_volume_module.shape  # (Z, Y, X)
        
        # Use the same conversion formula as used in processing (dx, dy)
        dx = float(self.pixel_size) / float(self.objective_magnification)  # meters per pixel
        dy = float(self.pixel_size) / float(self.objective_magnification)  # meters per pixel
        
        # Convert coordinates from micrometers to meters, then to pixels
        pos_x_m = pos_x_um / 1e6  # Convert µm to meters
        pos_y_m = pos_y_um / 1e6  # Convert µm to meters
        pos_z_m = pos_z_um / 1e6  # Convert µm to meters
        
        # Convert X,Y coordinates (camera pixel coordinates)
        pos_x_pix = int(pos_x_m / dx)
        pos_y_pix = int(pos_y_m / dy)
        
        # For Z coordinate, use the step parameter
        dz = float(self.step)  # meters
        pos_z_pix = int(pos_z_m / dz)
        
        # Clamp to valid range
        pos_x_pix = max(0, min(pos_x_pix, volume_shape[2] - 1))
        pos_y_pix = max(0, min(pos_y_pix, volume_shape[1] - 1))
        pos_z_pix = max(0, min(pos_z_pix, volume_shape[0] - 1))
        
        # print(f"📍 Corrected pixel coordinates: ({pos_x_pix}, {pos_y_pix}, {pos_z_pix})")
        # print(f"📐 Volume shape: {volume_shape}")
        # print(f"🔧 Spatial resolution: dx={dx*1e6:.3f} µm/pixel, dy={dy*1e6:.3f} µm/pixel")
        
        # Calculate effective pixel size for debugging
        effective_pixel_size = dx * 1e6  # Convert from meters to micrometers
        # print(f"🔍 Effective pixel size: {effective_pixel_size:.3f} µm/pixel")
        # Skip the problematic print line
        
        # Calculate slice boundaries with padding
        half_xy = vox_xy // 2
        half_z = vox_z // 2
        
        # XY slice (constant Z)
        z_center = max(0, min(pos_z_pix, volume_shape[0] - 1))
        xy_slice = self._extract_slice_with_padding(
            self.d_volume_module[z_center, :, :], 
            pos_y_pix, pos_x_pix, vox_xy, vox_xy
        )
        
        # XZ slice (constant Y) 
        y_center = max(0, min(pos_y_pix, volume_shape[1] - 1))
        xz_slice = self._extract_slice_with_padding(
            self.d_volume_module[:, y_center, :],
            pos_z_pix, pos_x_pix, vox_z, vox_xy
        )
        
        # YZ slice (constant X)
        x_center = max(0, min(pos_x_pix, volume_shape[2] - 1))
        yz_slice = self._extract_slice_with_padding(
            self.d_volume_module[:, :, x_center],
            pos_z_pix, pos_y_pix, vox_z, vox_xy
        )
        
        # print(f"✅ Extracted slices: XY({xy_slice.shape}), XZ({xz_slice.shape}), YZ({yz_slice.shape})")
        
        return {
            'xy_slice': cp.asnumpy(xy_slice),
            'xz_slice': cp.asnumpy(xz_slice),
            'yz_slice': cp.asnumpy(yz_slice)
        }

    def _extract_slice_with_padding(self, slice_2d, center_row, center_col, size_row, size_col):
        """Extract a sub-slice with zero padding if near boundaries"""
        half_row = size_row // 2
        half_col = size_col // 2
        
        # Calculate boundaries
        row_start = center_row - half_row
        row_end = center_row + half_row + (size_row % 2)
        col_start = center_col - half_col
        col_end = center_col + half_col + (size_col % 2)
        
        # Get original slice dimensions
        orig_rows, orig_cols = slice_2d.shape
        
        # Create output array filled with zeros
        output = cp.zeros((size_row, size_col), dtype=slice_2d.dtype)
        
        # Calculate valid regions
        valid_row_start = max(0, row_start)
        valid_row_end = min(orig_rows, row_end)
        valid_col_start = max(0, col_start)
        valid_col_end = min(orig_cols, col_end)
        
        # Calculate offsets in output array
        out_row_start = valid_row_start - row_start
        out_row_end = out_row_start + (valid_row_end - valid_row_start)
        out_col_start = valid_col_start - col_start
        out_col_end = out_col_start + (valid_col_end - valid_col_start)
        
        # Copy valid data
        output[out_row_start:out_row_end, out_col_start:out_col_end] = \
            slice_2d[valid_row_start:valid_row_end, valid_col_start:valid_col_end]
        
        return output

    def get_pixel_value(self, directory, filename, display_type, plane_number, x, y):
        """Get the pixel value from the original data for the given coordinates"""
        try:
            # Helper: get value from array with bounds checking
            def _get_array_value(arr, x, y):
                if hasattr(arr, 'shape') and len(arr.shape) >= 2:
                    h, w = arr.shape[-2:]
                    if 0 <= x < w and 0 <= y < h:
                        return arr[y, x] if len(arr.shape) == 2 else arr[-1, y, x]
                return None
            
            # RAW_HOLOGRAM
            if display_type == "RAW_HOLOGRAM":
                if self.h_raw_holo is not None:
                    return _get_array_value(self.h_raw_holo, x, y)
                # Fallback to loading file
                raw_hologram = self.open_hologram_image(directory, filename)
                arr = np.array(raw_hologram.convert('L'), dtype=np.float64)
                return _get_array_value(arr, x, y)
            
            # CLEANED_HOLOGRAM
            elif display_type == "CLEANED_HOLOGRAM":
                if self.h_cleaned_holo is not None:
                    return _get_array_value(self.h_cleaned_holo, x, y)
                    
            # FILTERED_HOLOGRAM
            elif display_type == "FILTERED_HOLOGRAM":
                # Try CPU version first (more efficient)
                if self.h_filtered_holo is not None:
                    return _get_array_value(self.h_filtered_holo, x, y)
                # Fallback to GPU version
                if self.d_filtered_holo is not None:
                    try:
                        h_filtered = cp.asnumpy(self.d_filtered_holo)
                        return _get_array_value(h_filtered, x, y)
                    except Exception:
                        pass
                        
            # FFT_HOLOGRAM
            elif display_type == "FFT_HOLOGRAM":
                if self.d_fft_holo is not None:
                    try:
                        h_fft = cp.asnumpy(self.d_fft_holo)
                        return _get_array_value(h_fft, x, y)
                    except Exception:
                        pass
                        
            # FFT_FILTERED_HOLOGRAM
            elif display_type == "FFT_FILTERED_HOLOGRAM":
                if self.d_fft_holo_filtered is not None:
                    try:
                        h_fft_filtered = cp.asnumpy(self.d_fft_holo_filtered)
                        return _get_array_value(h_fft_filtered, x, y)
                    except Exception:
                        pass
                        
            # VOLUME_PLANE_NUMBER
            elif display_type == "VOLUME_PLANE_NUMBER":
                if self.d_volume_module is not None:
                    try:
                        if plane_number < self.d_volume_module.shape[0]:
                            plane = cp.asnumpy(self.d_volume_module[plane_number, :, :])
                            return _get_array_value(plane, x, y)
                    except Exception:
                        pass
                        
            # Projections
            elif display_type in ["XY_SUM_PROJECTION", "XY_MAX_PROJECTION"]:
                if self.d_volume_module is not None:
                    try:
                        if display_type == "XY_SUM_PROJECTION":
                            projection = cp.sum(self.d_volume_module, axis=0)
                        else:
                            projection = cp.max(self.d_volume_module, axis=0)
                        projection = cp.asnumpy(projection)
                        return _get_array_value(projection, x, y)
                    except Exception:
                        pass
                        
            elif display_type in ["XZ_SUM_PROJECTION", "XZ_MAX_PROJECTION"]:
                if self.d_volume_module is not None:
                    try:
                        if display_type == "XZ_SUM_PROJECTION":
                            projection = cp.sum(self.d_volume_module, axis=1)
                        else:
                            projection = cp.max(self.d_volume_module, axis=1)
                        projection = cp.asnumpy(projection)
                        return _get_array_value(projection, x, y)
                    except Exception:
                        pass
                        
            elif display_type in ["YZ_SUM_PROJECTION", "YZ_MAX_PROJECTION"]:
                if self.d_volume_module is not None:
                    try:
                        if display_type == "YZ_SUM_PROJECTION":
                            projection = cp.sum(self.d_volume_module, axis=2)
                        else:
                            projection = cp.max(self.d_volume_module, axis=2)
                        projection = cp.asnumpy(projection)
                        return _get_array_value(projection, x, y)
                    except Exception:
                        pass
                        
            return "N/A"
            
        except Exception as e:
            return f"Error: {e}"

    def analyze_focus_at_position(self, x_pos, y_pos, focus_type_str, sum_size):
        """Analyzes focus function at a given position through all Z layers
        
        Args:
            x_pos: X position of the pixel to analyze
            y_pos: Y position of the pixel to analyze
            focus_type_str: Focus type ("TENEGRAD", "SUM_OF_INTENSITY", etc.)
            sum_size: Size of the summation window
            
        Returns:
            List of focus values for each Z plane or None in case of error
        """
        try:
            # Check that we have a propagated volume
            if self.d_volume_module is None:
                print("❌ No propagated volume available for focus analysis")
                return None
                
            # Check coordinates
            if (x_pos < 0 or x_pos >= self.d_volume_module.shape[2] or
                y_pos < 0 or y_pos >= self.d_volume_module.shape[1]):
                print(f"❌ Position ({x_pos}, {y_pos}) out of volume bounds")
                return None
                
            # Focus type mapping
            focus_type_map = {
                "SUM_OF_INTENSITY": Focus_type.SUM_OF_INTENSITY,
                "SUM_OF_LAPLACIAN": Focus_type.SUM_OF_LAPLACIAN,
                "SUM_OF_VARIANCE": Focus_type.SUM_OF_VARIANCE,
                "TENEGRAD": Focus_type.TENEGRAD
            }
            
            if focus_type_str not in focus_type_map:
                print(f"❌ Unknown focus type: {focus_type_str}")
                return None
                
            focus_type_enum = focus_type_map[focus_type_str]
            
            print(f"🔍 Focus analysis {focus_type_str} at position ({x_pos}, {y_pos}) with sum_size={sum_size}")
            
            # Create temporary volume for focus computation
            focus_volume = cp.copy(self.d_volume_module)
            
            # Apply focus function on entire volume
            focus.focus(focus_volume, focus_volume, sum_size, focus_type_enum)
            
            # Extract focus values for given position
            focus_values = []
            for z in range(focus_volume.shape[0]):
                # Extract value at specified pixel
                pixel_value = float(focus_volume[z, y_pos, x_pos])
                focus_values.append(pixel_value)
                
            print(f"✅ Analysis completed: {len(focus_values)} values extracted")
            
            # Clean up GPU memory
            del focus_volume
            cp.get_default_memory_pool().free_all_blocks()
            
            return focus_values
            
        except Exception as e:
            print(f"❌ Error during focus analysis: {e}")
            import traceback
            traceback.print_exc()
            return None
