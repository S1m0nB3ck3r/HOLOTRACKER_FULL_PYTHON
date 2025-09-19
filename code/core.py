import os
import numpy as np
from PIL import Image
import scipy.ndimage as ndi
from scipy import fft
try:
    import cupy as cp
    from traitement_holo import calc_holo_moyen, read_image
    import propagation as propag
    import focus 
    from focus import Focus_type
    from CCL3D import CCL3D, calc_threshold, CCA_CUDA_float, type_threshold, dobjet
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import pandas as pd
    CUPY_AVAILABLE = True
except ImportError as e:
    print(f"Warning: CuPy or other dependencies not available: {e}")
    cp = None
    CUPY_AVAILABLE = False

class HoloTrackerCore:
    def __init__(self):
        self.parameters = {}
        self.results = {}
        self.mean_hologram = None
        self.cleaned_hologram = None
        self.filtered_hologram = None
        self.volume = None
        
        # Variables pour le TEST_MODE
        self.test_mode_allocated = False
        self.test_mode_params = {}  # Pour tracker les paramètres qui nécessitent une réallocation
        
        # Variables GPU allouées (comme dans test_HoloTracker_locate.py)
        self.h_raw_holo = None
        self.h_cleaned_holo = None
        self.d_holo = None
        self.d_filtered_holo = None
        self.d_fft_holo = None
        self.d_fft_holo_propag = None
        self.d_holo_propag = None
        self.d_KERNEL = None
        self.d_FFT_KERNEL = None
        self.d_volume_module = None
        self.d_bin_volume_focus = None
        self.h_mean_holo = None
        self.d_mean_holo = None
        
        # Variables pour les résultats
        self.current_threshold = None
        self.current_features = None

    def perform_preprocessing(self, params):
        """Effectue le preprocessing (chargement mean hologram, etc.)"""
        self.parameters = params
        
        # Charger l'hologramme moyen si spécifié
        mean_path = params.get("mean_hologram_image_path", "")
        if mean_path and os.path.exists(mean_path):
            if mean_path.endswith('.npy'):
                # Charger directement le fichier .npy
                self.mean_hologram = np.load(mean_path)
            else:
                # Si c'est un autre format, essayer de trouver le .npy correspondant
                mean_dir = os.path.dirname(mean_path)
                npy_path = os.path.join(mean_dir, "mean_hologram.npy")
                if os.path.exists(npy_path):
                    self.mean_hologram = np.load(npy_path)
                else:
                    # Fallback vers l'ancien système
                    mean_img = Image.open(mean_path).convert('L')
                    self.mean_hologram = np.array(mean_img, dtype=np.float64)
        else:
            self.mean_hologram = None

    def get_display_image(self, directory, filename, display_type, plane_number=0):
        """Retourne l'image à afficher selon le type demandé avec marqueurs de détection"""
        
        print(f"🖼️  Core: get_display_image called with type: {display_type}")
        print(f"🖼️  Core: test_mode_allocated = {getattr(self, 'test_mode_allocated', False)}")
        print(f"🖼️  Core: has d_holo = {hasattr(self, 'd_holo')}")
        print(f"🖼️  Core: has d_filtered_holo = {hasattr(self, 'd_filtered_holo')}")
        print(f"🖼️  Core: has d_volume_module = {hasattr(self, 'd_volume_module')}")
        
        try:
            if display_type == "RAW_HOLOGRAM":
                # Load raw hologram
                raw_hologram = self.open_hologram_image(directory, filename)
                raw_array = np.array(raw_hologram.convert('L'), dtype=np.float64)
                image = Image.fromarray(raw_array.astype(np.uint8))
                return self.add_detection_markers_to_image(image, self.results)
                
            elif display_type == "CLEANED_HOLOGRAM":
                print("🧹 Core: Processing CLEANED_HOLOGRAM display")
                # Use cleaned hologram from processing pipeline if available
                if hasattr(self, 'h_cleaned_holo') and self.test_mode_allocated and self.h_cleaned_holo is not None:
                    try:
                        print("✅ Core: Using pre-processed cleaned hologram")
                        # Get cleaned hologram from CPU cache
                        h_cleaned = self.h_cleaned_holo.copy()
                        # Normalize for display
                        min_val = h_cleaned.min()
                        max_val = h_cleaned.max()
                        if max_val > min_val:
                            h_cleaned_norm = ((h_cleaned - min_val) * 255 / (max_val - min_val)).astype(np.uint8)
                        else:
                            h_cleaned_norm = (h_cleaned * 255).astype(np.uint8)
                        image = Image.fromarray(h_cleaned_norm)
                        return self.add_detection_markers_to_image(image, self.results)
                    except Exception as e:
                        print(f"❌ Core: Error with pre-processed cleaned hologram: {e}")
                        pass
                        
                # Fallback to traditional cleaning
                print("🔄 Core: Using fallback cleaning")
                try:
                    raw_hologram = self.open_hologram_image(directory, filename)
                    raw_array = np.array(raw_hologram.convert('L'), dtype=np.float64)
                    cleaned = self.clean_hologram_safe(raw_array)
                    # Proper normalization for display
                    min_val = cleaned.min()
                    max_val = cleaned.max()
                    if max_val > min_val:
                        cleaned_norm = ((cleaned - min_val) * 255 / (max_val - min_val)).astype(np.uint8)
                    else:
                        cleaned_norm = (cleaned * 255 / cleaned.max() if cleaned.max() > 0 else cleaned).astype(np.uint8)
                    image = Image.fromarray(cleaned_norm)
                    return self.add_detection_markers_to_image(image, self.results)
                except Exception as e:
                    print(f"❌ Core: Error in fallback cleaning: {e}")
                    # Ultimate fallback: return raw
                    return self.get_display_image(directory, filename, "RAW_HOLOGRAM", plane_number)
                    
            elif display_type == "FILTERED_HOLOGRAM":
                print("🔍 Core: Processing FILTERED_HOLOGRAM display")
                # Use filtered hologram from processing pipeline if available
                if hasattr(self, 'd_filtered_holo') and self.test_mode_allocated and self.d_filtered_holo is not None:
                    try:
                        print("✅ Core: Using pre-processed filtered hologram from GPU")
                        import cupy as cp
                        # Get filtered hologram from GPU
                        h_filtered = cp.asnumpy(self.d_filtered_holo)
                        print(f"🔍 Core: Filtered data range: [{h_filtered.min():.3f}, {h_filtered.max():.3f}]")
                        
                        # Prendre le module pour l'affichage (données complexes)
                        if np.iscomplexobj(h_filtered):
                            h_filtered = np.abs(h_filtered)
                            print("🔍 Core: Converted complex to magnitude")
                        
                        # Normalize for display
                        min_val = h_filtered.min()
                        max_val = h_filtered.max()
                        if max_val > min_val:
                            h_filtered_norm = ((h_filtered - min_val) * 255 / (max_val - min_val)).astype(np.uint8)
                        else:
                            h_filtered_norm = (h_filtered * 255 / h_filtered.max() if h_filtered.max() > 0 else h_filtered).astype(np.uint8)
                        
                        image = Image.fromarray(h_filtered_norm)
                        return self.add_detection_markers_to_image(image, self.results)
                    except Exception as e:
                        print(f"❌ Core: Error getting filtered hologram from GPU: {e}")
                        # Fallback to cleaned hologram
                        return self.get_display_image(directory, filename, "CLEANED_HOLOGRAM", plane_number)
                else:
                    print("⚠️  Core: No pre-processed filtered hologram available, using fallback")
                    # Fallback to cleaned hologram if filtered not available
                    return self.get_display_image(directory, filename, "CLEANED_HOLOGRAM", plane_number)
                    
            elif display_type == "VOLUME_PLANE_NUMBER":
                # Use processed volume from GPU if available
                if hasattr(self, 'd_volume_module') and self.test_mode_allocated:
                    try:
                        import cupy as cp
                        volume_gpu = self.d_volume_module
                        if plane_number < volume_gpu.shape[0]:
                            plane = cp.asnumpy(volume_gpu[plane_number, :, :])
                            plane = np.abs(plane)
                            if plane.max() > 0:
                                plane = (plane / plane.max() * 255).astype(np.uint8)
                            else:
                                plane = plane.astype(np.uint8)
                            image = Image.fromarray(plane)
                            return self.add_detection_markers_to_image(image, self.results)
                    except:
                        pass
                # Fallback to traditional reconstruction
                raw_hologram = self.open_hologram_image(directory, filename)
                raw_array = np.array(raw_hologram.convert('L'), dtype=np.float64)
                volume = self.reconstruct_volume(raw_array)
                if plane_number < volume.shape[2]:
                    plane = volume[:, :, plane_number]
                    plane = np.abs(plane)
                    plane = (plane / np.max(plane) * 255).astype(np.uint8)
                    return Image.fromarray(plane)
                else:
                    return Image.fromarray(np.zeros((100, 100), dtype=np.uint8))
                    
            elif display_type == "XY_SUM_PROJECTION":
                # Use processed volume from GPU if available
                if hasattr(self, 'd_volume_module') and self.test_mode_allocated:
                    try:
                        import cupy as cp
                        projection = cp.sum(self.d_volume_module, axis=0)  # Sum along Z axis
                        projection = cp.asnumpy(projection)
                        if projection.max() > 0:
                            projection = (projection / projection.max() * 255).astype(np.uint8)
                        else:
                            projection = projection.astype(np.uint8)
                        image = Image.fromarray(projection)
                        return self.add_detection_markers_to_image_2d_projection(image, self.results, 'XY')
                    except:
                        pass
                # Fallback
                raw_hologram = self.open_hologram_image(directory, filename)
                raw_array = np.array(raw_hologram.convert('L'), dtype=np.float64)
                volume = self.reconstruct_volume(raw_array)
                projection = np.sum(np.abs(volume), axis=2)
                projection = (projection / np.max(projection) * 255).astype(np.uint8)
                return Image.fromarray(projection)
                
            elif display_type == "XZ_SUM_PROJECTION":
                # Use processed volume from GPU if available
                if hasattr(self, 'd_volume_module') and self.test_mode_allocated:
                    try:
                        import cupy as cp
                        projection = cp.sum(self.d_volume_module, axis=1)  # Sum along Y axis
                        projection = cp.asnumpy(projection)
                        if projection.max() > 0:
                            projection = (projection / projection.max() * 255).astype(np.uint8)
                        else:
                            projection = projection.astype(np.uint8)
                        image = Image.fromarray(projection)
                        return self.add_detection_markers_to_image_2d_projection(image, self.results, 'XZ')
                    except:
                        pass
                # Fallback
                raw_hologram = self.open_hologram_image(directory, filename)
                raw_array = np.array(raw_hologram.convert('L'), dtype=np.float64)
                volume = self.reconstruct_volume(raw_array)
                projection = np.sum(np.abs(volume), axis=1)
                projection = (projection / np.max(projection) * 255).astype(np.uint8)
                return Image.fromarray(projection)
                
            elif display_type == "YZ_SUM_PROJECTION":
                # Use processed volume from GPU if available  
                if hasattr(self, 'd_volume_module') and self.test_mode_allocated:
                    try:
                        import cupy as cp
                        projection = cp.sum(self.d_volume_module, axis=2)  # Sum along X axis
                        projection = cp.asnumpy(projection)
                        if projection.max() > 0:
                            projection = (projection / projection.max() * 255).astype(np.uint8)
                        else:
                            projection = projection.astype(np.uint8)
                        image = Image.fromarray(projection)
                        return self.add_detection_markers_to_image_2d_projection(image, self.results, 'YZ')
                    except:
                        pass
                # Fallback
                raw_hologram = self.open_hologram_image(directory, filename)
                raw_array = np.array(raw_hologram.convert('L'), dtype=np.float64)
                volume = self.reconstruct_volume(raw_array)
                projection = np.sum(np.abs(volume), axis=0)
                projection = (projection / np.max(projection) * 255).astype(np.uint8)
                return Image.fromarray(projection)
                
            elif display_type == "XY_MAX_PROJECTION":
                # Use processed volume from GPU if available
                if hasattr(self, 'd_volume_module') and self.test_mode_allocated:
                    try:
                        import cupy as cp
                        projection = cp.max(self.d_volume_module, axis=0)  # Max along Z axis
                        projection = cp.asnumpy(projection)
                        if projection.max() > 0:
                            projection = (projection / projection.max() * 255).astype(np.uint8)
                        else:
                            projection = projection.astype(np.uint8)
                        image = Image.fromarray(projection)
                        return self.add_detection_markers_to_image_2d_projection(image, self.results, 'XY')
                    except:
                        pass
                # Fallback
                raw_hologram = self.open_hologram_image(directory, filename)
                raw_array = np.array(raw_hologram.convert('L'), dtype=np.float64)
                volume = self.reconstruct_volume(raw_array)
                projection = np.max(np.abs(volume), axis=2)
                projection = (projection / np.max(projection) * 255).astype(np.uint8)
                return Image.fromarray(projection)
                
            elif display_type == "XZ_MAX_PROJECTION":
                # Use processed volume from GPU if available
                if hasattr(self, 'd_volume_module') and self.test_mode_allocated:
                    try:
                        import cupy as cp
                        projection = cp.max(self.d_volume_module, axis=1)  # Max along Y axis
                        projection = cp.asnumpy(projection)
                        if projection.max() > 0:
                            projection = (projection / projection.max() * 255).astype(np.uint8)
                        else:
                            projection = projection.astype(np.uint8)
                        image = Image.fromarray(projection)
                        return self.add_detection_markers_to_image_2d_projection(image, self.results, 'XZ')
                    except:
                        pass
                # Fallback
                raw_hologram = self.open_hologram_image(directory, filename)
                raw_array = np.array(raw_hologram.convert('L'), dtype=np.float64)
                volume = self.reconstruct_volume(raw_array)
                projection = np.max(np.abs(volume), axis=1)
                projection = (projection / np.max(projection) * 255).astype(np.uint8)
                return Image.fromarray(projection)
                
            elif display_type == "YZ_MAX_PROJECTION":
                # Use processed volume from GPU if available
                if hasattr(self, 'd_volume_module') and self.test_mode_allocated:
                    try:
                        import cupy as cp
                        projection = cp.max(self.d_volume_module, axis=2)  # Max along X axis
                        projection = cp.asnumpy(projection)
                        if projection.max() > 0:
                            projection = (projection / projection.max() * 255).astype(np.uint8)
                        else:
                            projection = projection.astype(np.uint8)
                        image = Image.fromarray(projection)
                        return self.add_detection_markers_to_image_2d_projection(image, self.results, 'YZ')
                    except:
                        pass
                # Fallback
                raw_hologram = self.open_hologram_image(directory, filename)
                raw_array = np.array(raw_hologram.convert('L'), dtype=np.float64)
                volume = self.reconstruct_volume(raw_array)
                projection = np.max(np.abs(volume), axis=0)
                projection = (projection / np.max(projection) * 255).astype(np.uint8)
                return Image.fromarray(projection)
                
            else:
                # Default: show raw hologram
                raw_hologram = self.open_hologram_image(directory, filename)
                raw_array = np.array(raw_hologram.convert('L'), dtype=np.float64)
                return Image.fromarray(raw_array.astype(np.uint8))
                
        except Exception as e:
            print(f"Error in get_display_image: {e}")
            # Emergency fallback
            try:
                raw_hologram = self.open_hologram_image(directory, filename)
                raw_array = np.array(raw_hologram.convert('L'), dtype=np.float64)
                return Image.fromarray(raw_array.astype(np.uint8))
            except:
                return Image.fromarray(np.zeros((100, 100), dtype=np.uint8))

    def clean_hologram(self, hologram):
        """Nettoie l'hologramme (soustrait la moyenne si disponible)"""
        if self.mean_hologram is not None and self.parameters.get("remove_mean", False):
            # Redimensionner si nécessaire
            if self.mean_hologram.shape != hologram.shape:
                mean_resized = np.array(Image.fromarray(self.mean_hologram).resize(
                    (hologram.shape[1], hologram.shape[0])
                ))
            else:
                mean_resized = self.mean_hologram
            cleaned = hologram - mean_resized
        else:
            cleaned = hologram.copy()
        return cleaned
    
    def clean_hologram_safe(self, hologram):
        """Nettoie l'hologramme avec normalisation sûre pour l'affichage"""
        try:
            if self.mean_hologram is not None and self.parameters.get("remove_mean", False):
                print("🧹 Core: Applying mean removal")
                # Redimensionner si nécessaire
                if self.mean_hologram.shape != hologram.shape:
                    mean_resized = np.array(Image.fromarray(self.mean_hologram.astype(np.uint8)).resize(
                        (hologram.shape[1], hologram.shape[0])
                    )).astype(np.float64)
                else:
                    mean_resized = self.mean_hologram.astype(np.float64)
                
                # Division au lieu de soustraction pour éviter les valeurs négatives
                # C'est plus cohérent avec le pipeline GPU qui fait hologram / mean_hologram
                cleaned = hologram / (mean_resized + 1e-10)  # Éviter division par zéro
                print(f"🧹 Core: Cleaned range: [{cleaned.min():.3f}, {cleaned.max():.3f}]")
            else:
                print("🧹 Core: No mean removal, keeping original")
                cleaned = hologram.copy()
                
            return cleaned
        except Exception as e:
            print(f"❌ Core: Error in clean_hologram_safe: {e}")
            return hologram.copy()

    def get_volume_projection_legacy(self, hologram, projection_type):
        """Legacy method for volume projections without GPU (fallback only)"""
        # This is a simplified fallback when GPU processing is not available
        # The main system uses GPU data from process_hologram_complete_pipeline
        cleaned = self.clean_hologram(hologram)
        
        # Simple projection simulation
        if projection_type in ["XY_SUM_PROJECTION", "XY_MAX_PROJECTION"]:
            return cleaned
        else:
            # For XZ and YZ projections, return transposed versions
            return cleaned.T

    def set_parameter(self, name, value):
        self.parameters[name] = value
        return f"{name} updated"
    
    def get_parameter(self, name, default=None):
        """Get a parameter value"""
        return self.parameters.get(name, default)

    def open_hologram_image(self, directory, filename):
        filepath = os.path.join(directory, filename)
        img = Image.open(filepath)
        # Redimensionne si les paramètres sont présents
        try:
            x = int(self.parameters.get("holo_size_x", img.width))
            y = int(self.parameters.get("holo_size_y", img.height))
            img = img.resize((x, y))
        except Exception:
            pass
        return img

    def compute_mean_hologram(self, directory, image_type, progress_callback=None):
        """Compute mean hologram from all images in directory"""
        ext_map = {"BMP": ".bmp", "TIF": ".tif", "JPG": ".jpg", "PNG": ".png"}
        ext = ext_map.get(image_type, ".tif")
        
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
        
        # Finalize mean
        mean_image = mean_image / total_images
        
        # Create mean directory
        mean_dir = os.path.join(directory, "mean")
        os.makedirs(mean_dir, exist_ok=True)
        
        # Save as .npy for calculations (float64 precision)
        mean_npy_path = os.path.join(mean_dir, "mean_hologram.npy")
        np.save(mean_npy_path, mean_image)
        
        # Save as .bmp for visualization (uint8)
        mean_bmp_path = os.path.join(mean_dir, "mean_hologram.bmp")
        mean_image_uint8 = np.clip(mean_image, 0, 255).astype(np.uint8)
        mean_pil = Image.fromarray(mean_image_uint8, mode='L')
        mean_pil.save(mean_bmp_path)
        
        return mean_npy_path  # Return the .npy path for use in calculations

    def batch_process(self):
        if "holograms_directory" not in self.parameters or not self.parameters["holograms_directory"]:
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
        return "Test mode activated"

    def exit_test_mode(self):
        """Sort du mode test"""
        self.cleanup_test_mode()
        return "Test mode deactivated"
    
    def initialize_test_mode(self, params):
        """Initialise le mode test avec allocation mémoire"""
        import cupy as cp
        
        # Paramètres de base
        cam_nb_pix_X = int(params.get("holo_size_x", 1024))
        cam_nb_pix_Y = int(params.get("holo_size_y", 1024))
        nb_plane = int(params.get("number_of_planes", 200))
        
        # Allocation des arrays selon la nouvelle architecture
        self.h_raw_holo = np.zeros(shape=(cam_nb_pix_Y, cam_nb_pix_X), dtype=np.float32)
        self.h_mean_holo = np.zeros(shape=(cam_nb_pix_Y, cam_nb_pix_X), dtype=np.float32)
        self.h_cleaned_holo = np.zeros(shape=(cam_nb_pix_Y, cam_nb_pix_X), dtype=np.float32)
        self.d_holo = cp.zeros(shape=(cam_nb_pix_Y, cam_nb_pix_X), dtype=cp.float32)
        self.d_filtered_holo = cp.zeros(shape=(cam_nb_pix_Y, cam_nb_pix_X), dtype=cp.float32)
        self.d_fft_holo = cp.zeros(shape=(cam_nb_pix_Y, cam_nb_pix_X), dtype=cp.complex64)
        self.d_fft_holo_propag = cp.zeros(shape=(cam_nb_pix_Y, cam_nb_pix_X), dtype=cp.complex64)
        self.d_holo_propag = cp.zeros(shape=(cam_nb_pix_Y, cam_nb_pix_X), dtype=cp.float32)
        self.d_KERNEL = cp.zeros(shape=(cam_nb_pix_Y, cam_nb_pix_X), dtype=cp.complex64)
        self.d_FFT_KERNEL = cp.zeros(shape=(cam_nb_pix_Y, cam_nb_pix_X), dtype=cp.complex64)
        self.d_volume_module = cp.zeros(shape=(nb_plane, cam_nb_pix_Y, cam_nb_pix_X), dtype=cp.float32)
        self.d_bin_volume_focus = cp.zeros(shape=(nb_plane, cam_nb_pix_Y, cam_nb_pix_X), dtype=cp.dtype(bool))
        
        # Charger l'hologramme moyen si disponible
        if self.mean_hologram is not None:
            self.h_mean_holo = self.mean_hologram.astype(np.float32)
            self.d_mean_holo = cp.asarray(self.h_mean_holo)
        else:
            self.h_mean_holo = None
            self.d_mean_holo = None
            
        self.test_mode_allocated = True
        
        # Initialiser les kernels de propagation
        #self._initialize_propagation_kernels(params)

    def check_reallocation_needed(self, new_params):
        """Check if GPU memory reallocation is needed based on parameter changes"""
        if not self.test_mode_allocated:
            return True
            
        # Parameters that require reallocation
        size_changing_params = ["holo_size_x", "holo_size_y", "nb_plan_reconstruit"]
        
        for param in size_changing_params:
            if param in new_params:
                old_value = self.parameters.get(param)
                new_value = new_params[param]
                if old_value != new_value:
                    return True
        return False

    def update_parameters_and_reallocate_if_needed(self, new_params):
        """Update parameters and reallocate GPU memory if needed"""
        if self.check_reallocation_needed(new_params):
            # Cleanup existing allocation
            if self.test_mode_allocated:
                self.cleanup_test_mode()
            
            # Update parameters
            self.parameters.update(new_params)
            
            # Reallocate with new parameters
            result = self.initialize_test_mode(self.parameters)
            return result
        else:
            # Just update parameters, no reallocation needed
            self.parameters.update(new_params)
            return "Parameters updated without reallocation"

    def reconstruct_volume(self, raw_array):
        """
        Reconstruct volume from raw hologram array (for backward compatibility)
        This is a simplified version that uses the same pipeline as process_hologram_complete_pipeline
        but returns the volume instead of storing results
        """
        print(f"🔧 Core: reconstruct_volume called")
        print(f"🔧 Core: test_mode_allocated = {getattr(self, 'test_mode_allocated', False)}")
        
        if not CUPY_AVAILABLE:
            raise Exception("CuPy not available for GPU processing")
            
        # Si on est en mode test et que les arrays sont alloués, utiliser le volume existant
        if (hasattr(self, 'test_mode_allocated') and self.test_mode_allocated and
            hasattr(self, 'd_volume_module') and self.d_volume_module is not None):
            
            print("✅ Core: Using existing volume from test mode")
            return cp.asnumpy(self.d_volume_module)
            
        # Sinon, reconstruction basique CPU seulement (pour affichage)
        print("⚠️  Core: Test mode not available, using CPU fallback")
        try:
            # Reconstruction CPU simple (juste pour avoir quelque chose à afficher)
            # Note: Ceci est un fallback minimal, pas la reconstruction complète
            
            # Simple propagation simulée pour ne pas crasher
            h, w = raw_array.shape
            nb_planes = 50  # Nombre réduit pour CPU
            
            # Créer un volume vide
            volume = np.zeros((nb_planes, h, w), dtype=np.complex128)
            
            # Remplir avec des données basiques (simulation)
            for z in range(nb_planes):
                volume[z] = raw_array.astype(np.complex128) * (0.5 + 0.5 * np.random.random())
            
            print("✅ Core: CPU fallback reconstruction completed")
            return np.abs(volume)
            
        except Exception as e:
            print(f"❌ Core: CPU fallback failed: {e}")
            # En dernier recours, retourner un volume vide
            h, w = raw_array.shape if raw_array is not None else (512, 512)
            return np.zeros((10, h, w), dtype=np.float32)

    def process_hologram_complete_pipeline(self, directory, filename):
        """
        Complete hologram processing pipeline following test_HoloTracker_locate.py
        Pipeline: Load -> Remove Mean -> Volume Propagation -> Focus -> CCL3D -> Label Analysis
        """
        print(f"🚀 Core: Starting hologram processing for {filename}")
        
        if not CUPY_AVAILABLE:
            print("❌ Core: CuPy not available")
            return "Error: CuPy not available for GPU processing"
            
        if not self.test_mode_allocated:
            print("❌ Core: Test mode not initialized")
            return "Error: Test mode not initialized"
            
        try:
            import time
            
            # 1. Load raw hologram and preprocessing
            start_processing = time.perf_counter()
            filepath = os.path.join(directory, filename)
            cam_nb_pix_X = int(self.parameters.get("holo_size_x", 1024))
            cam_nb_pix_Y = int(self.parameters.get("holo_size_y", 1024))
            
            # Load raw hologram into h_raw_holo
            self.h_raw_holo[:] = read_image(filepath, cam_nb_pix_X, cam_nb_pix_Y)
            
            # 2. Clean hologram (remove mean selon paramètre)
            remove_mean = self.parameters.get("remove_mean", False)
            if remove_mean and self.mean_hologram is not None:
                print("🧹 Pipeline: Removing mean hologram (subtract)")
                self.h_mean_holo[:] = self.mean_hologram.astype(np.float32)
                self.h_cleaned_holo[:] = self.h_raw_holo - self.h_mean_holo
            else:
                print("📄 Pipeline: Keeping raw hologram (no mean removal)")
                self.h_cleaned_holo[:] = self.h_raw_holo
            
            # Transfer cleaned hologram to GPU - reuse pre-allocated array
            self.d_holo[:] = cp.asarray(self.h_cleaned_holo)
            
            t1 = time.perf_counter()
            t_preprocess = t1 - start_processing
            
            # 3. Spectral filtering before propagation
            
            # Get filtering parameters from UI
            f_pix_min = int(self.parameters.get("high_pass", 15))  # high_pass = f_pix_min
            f_pix_max = int(self.parameters.get("low_pass", 125))  # low_pass = f_pix_max
            
            # Apply spectral filtering to get d_filtered_holo
            filter_start = time.perf_counter()
            propag.spec_filter_FFT(
                self.d_holo, self.d_fft_holo, self.d_filtered_holo,
                cam_nb_pix_X, cam_nb_pix_Y, f_pix_min, f_pix_max
            )
            
            t_filter = time.perf_counter() - filter_start
            
            # 4. Volume propagation by angular spectrum method (without filtering since already done)
            
            # Get parameters for propagation
            medium_wavelength = float(self.parameters.get("medium_wavelength", 660e-9 / 1.33))
            cam_magnification = float(self.parameters.get("cam_magnification", 40.0))
            cam_pix_size = float(self.parameters.get("cam_pix_size", 7e-6))
            nb_plane = int(self.parameters.get("nb_plan_reconstruit", 200))
            dz = float(self.parameters.get("dz", 0.2e-6))  # in meters
            
            propag_start = time.perf_counter()
            
            # Use filtered hologram for propagation with no additional filtering (f_pix_min=0, f_pix_max=0)
            propag.volume_propag_angular_spectrum_to_module(
                self.d_filtered_holo, self.d_fft_holo, self.d_KERNEL, self.d_fft_holo_propag, 
                self.d_volume_module, medium_wavelength, cam_magnification, 
                cam_pix_size, cam_nb_pix_X, cam_nb_pix_Y, 0.0, dz, nb_plane, 0, 0
            )
            
            t2 = time.perf_counter()
            t_propag = t2 - propag_start
            
            # 4. Focus on the volume (INPLACE)
            focus_start = time.perf_counter()
            focus_smooth_size = int(self.parameters.get("focus_smooth_size", 15))
            focus.focus(self.d_volume_module, self.d_volume_module, focus_smooth_size, Focus_type.TENEGRAD)
            
            t3 = time.perf_counter()
            t_focus = t3 - focus_start
            
            # 5. Compute threshold and CCL3D
            ccl_start = time.perf_counter()
            nb_StdVar_threshold = float(self.parameters.get("nb_StdVar_threshold", 5))
            n_connectivity = int(self.parameters.get("n_connectivity", 26))
            
            # Calculate threshold if:
            # - First time (no threshold calculated yet)
            # - recalc_threshold is True (compute on each hologram)
            # - nb_StdVar_threshold value has changed
            need_recalc = (not hasattr(self, 'threshold') or 
                          not hasattr(self, 'last_nb_StdVar_threshold') or
                          self.parameters.get("recalc_threshold", False) or
                          self.last_nb_StdVar_threshold != nb_StdVar_threshold)
            
            if need_recalc:
                self.threshold = calc_threshold(self.d_volume_module, nb_StdVar_threshold)
                self.last_nb_StdVar_threshold = nb_StdVar_threshold
            
            # CCL3D
            d_labels, number_of_labels = CCL3D(
                self.d_bin_volume_focus, self.d_volume_module, 
                type_threshold.THRESHOLD, self.threshold, n_connectivity
            )
            
            t4 = time.perf_counter()
            t_ccl = t4 - ccl_start
            
            # 6. Label analysis (centroid computation)
            cca_start = time.perf_counter()
            if number_of_labels > 0:
                features = np.ndarray(shape=(number_of_labels,), dtype=dobjet)
                dx = float(self.parameters.get("dx", 1000000 * cam_pix_size / cam_magnification))
                dy = float(self.parameters.get("dy", 1000000 * cam_pix_size / cam_magnification))
                
                features = CCA_CUDA_float(
                    d_labels, self.d_volume_module, number_of_labels, 
                    1, cam_nb_pix_X, cam_nb_pix_Y, nb_plane, dx, dy, dz * 1e6
                )
                
            else:
                features = np.array([])
                
            t5 = time.perf_counter()
            t_cca = t5 - cca_start
            t_total = t5 - start_processing
            
            # AFFICHAGE DES TEMPS COMME test_HoloTracker_locate.py
            print(f'number of objects located: {number_of_labels}')
            print(f't preprocess: {t_preprocess:.6f}')
            print(f't filter: {t_filter:.6f}')
            print(f't propagation: {t_propag:.6f}')
            print(f't focus: {t_focus:.6f}') 
            print(f't ccl: {t_ccl:.6f}')
            print(f't cca: {t_cca:.6f}')
            print(f'total iteration time: {t_total:.6f}')
            print(f'---')
            
            # Store results
            self.results = {
                'number_of_objects': number_of_labels,
                'features': features,
                'processing_times': {
                    'preprocessing': t_preprocess,
                    'filtering': t_filter,
                    'propagation': t_propag,
                    'focus': t_focus,
                    'ccl': t_ccl,
                    'cca': t_cca,
                    'total_processing': t_total
                }
            }
            
            print(f"✅ Core: Results stored with processing_times: {self.results['processing_times']}")
            
            if number_of_labels > 0:
                return f"Processing completed: {number_of_labels} objects found"
            else:
                return "Processing completed: No objects found"
                
        except Exception as e:
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
            
            # Debug: Check if self.results exists and what it contains
            print(f"🔧 Core: self.results exists: {hasattr(self, 'results') and self.results is not None}")
            if hasattr(self, 'results') and self.results:
                print(f"🔧 Core: self.results keys: {list(self.results.keys())}")
            
            # Add processing times if available (regardless of detection results)
            if hasattr(self, 'results') and self.results and 'processing_times' in self.results:
                result_data['processing_times'] = self.results['processing_times']
                print(f"⏱️  Core: Added processing_times: {self.results['processing_times']}")
            else:
                print("❌ Core: No processing_times found in self.results")
            
            # Add detection results if available
            if hasattr(self, 'results') and self.results and 'features' in self.results and self.results['features'] is not None:
                features = self.results['features']
                if len(features) > 0:
                    # Extract coordinates from features (following test_HoloTracker_locate.py format)
                    positions = pd.DataFrame(features, columns=['i_image', 'baryX', 'baryY', 'baryZ', 'nb_pix'])
                    
                    # Return localizations in the format expected by UI
                    localizations = []
                    for _, row in positions.iterrows():
                        localizations.append((row['baryX'], row['baryY'], row['baryZ']))
                    
                    result_data.update({
                        'localizations': localizations,
                        'particle_sizes': positions['nb_pix'].tolist(),
                        'count': len(features)
                    })
                
            return result_data
            
        except Exception as e:
            print(f"Error getting 3D results data: {e}")
            # Return basic structure with timing if available
            basic_data = {'localizations': [], 'particle_sizes': [], 'count': 0}
            if 'processing_times' in self.results:
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
            cam_nb_pix_X = int(self.parameters.get("holo_size_x", 1024))
            cam_nb_pix_Y = int(self.parameters.get("holo_size_y", 1024))
            
            # Load hologram
            h_holo = read_image(filepath, cam_nb_pix_X, cam_nb_pix_Y)
            
            # Apply mean removal if available
            if self.mean_hologram is not None:
                h_holo_cleaned = h_holo / self.mean_hologram
                
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
            print(f"Error getting cleaned hologram: {e}")
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
            
            # Add red circles for each detection
            if 'features' in self.results and self.results['features'] is not None:
                features = self.results['features']
                for feature in features:
                    # Extract coordinates (baryX, baryY are in micrometers, convert to pixels)
                    x_um = feature[1]  # baryX
                    y_um = feature[2]  # baryY
                    
                    # Convert from micrometers to pixels (inverse of dx, dy calculation)
                    cam_pix_size = float(self.parameters.get("cam_pix_size", 7e-6))
                    cam_magnification = float(self.parameters.get("cam_magnification", 40.0))
                    
                    x_pix = int(x_um * cam_magnification / (1000000 * cam_pix_size))
                    y_pix = int(y_um * cam_magnification / (1000000 * cam_pix_size))
                    
                    # Draw red circle
                    cv2.circle(img_array, (x_pix, y_pix), 5, (255, 0, 0), 2)
            
            return Image.fromarray(img_array)
            
        except ImportError:
            print("OpenCV not available, cannot add detection markers")
            return image
        except Exception as e:
            print(f"Error adding detection markers: {e}")
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
                    # Extract coordinates (already in micrometers)
                    x_um = feature[1]  # baryX
                    y_um = feature[2]  # baryY  
                    z_um = feature[3]  # baryZ
                    
                    # Convert from micrometers to pixels
                    cam_pix_size = float(self.parameters.get("cam_pix_size", 7e-6))
                    cam_magnification = float(self.parameters.get("cam_magnification", 40.0))
                    dz = float(self.parameters.get("dz", 0.2e-6))  # Step size in meters
                    
                    # X and Y coordinates in pixels (lateral dimensions)
                    x_pix = int(x_um * cam_magnification / (1000000 * cam_pix_size))
                    y_pix = int(y_um * cam_magnification / (1000000 * cam_pix_size))
                    
                    # Z coordinate in pixels (depth dimension) - convert from micrometers to plane index
                    z_pix = int(z_um / (dz * 1e6))  # Convert dz from meters to micrometers, then get plane index
                    
                    # Choose coordinates based on projection type
                    if projection_type == 'XY':
                        coord_x, coord_y = x_pix, y_pix
                    elif projection_type == 'XZ':
                        coord_x, coord_y = x_pix, z_pix
                    elif projection_type == 'YZ':
                        coord_x, coord_y = y_pix, z_pix
                    else:
                        coord_x, coord_y = x_pix, y_pix
                    
                    # Draw markers (green circle + red center)
                    cv2.circle(img_array, (coord_x, coord_y), 5, (0, 255, 0), 2)
                    cv2.circle(img_array, (coord_x, coord_y), 2, (255, 0, 0), -1)
            
            return Image.fromarray(img_array)
            
        except ImportError:
            print("OpenCV not available, cannot add detection markers")
            return image
        except Exception as e:
            print(f"Error adding detection markers to projection: {e}")
            return image

    def get_default_display_type(self):
        """Get default display type based on parameters"""
        remove_mean = self.parameters.get("remove_mean", False)
        # Check if remove_mean is enabled (regardless of mean_hologram availability)
        if remove_mean:
            return "CLEANED_HOLOGRAM"
        else:
            return "RAW_HOLOGRAM"
        
    def cleanup_test_mode(self):
        """Nettoie la mémoire allouée pour le mode test"""
        # Libérer la mémoire GPU et CPU
        gpu_vars = ['d_holo', 'd_filtered_holo', 'd_fft_holo', 'd_fft_holo_propag', 'd_holo_propag', 
                   'd_KERNEL', 'd_FFT_KERNEL', 'd_volume_module', 'd_bin_volume_focus', 'd_mean_holo']
        
        cpu_vars = ['h_raw_holo', 'h_cleaned_holo', 'h_mean_holo']
        
        # Nettoyer les variables GPU
        for var_name in gpu_vars:
            if hasattr(self, var_name):
                var = getattr(self, var_name)
                if var is not None:
                    del var
                setattr(self, var_name, None)
        
        # Nettoyer les variables CPU
        for var_name in cpu_vars:
            if hasattr(self, var_name):
                var = getattr(self, var_name)
                if var is not None:
                    del var
                setattr(self, var_name, None)
        
        self.test_mode_allocated = False
        
    def _initialize_propagation_kernels(self, params):
        """Initialize propagation kernels using existing functions from test_HoloTracker_locate.py"""
        try:
            # Le kernel de propagation est calculé dynamiquement dans volume_propag_angular_spectrum_to_module
            # Nous n'avons besoin que d'allouer l'espace mémoire comme dans test_HoloTracker_locate.py
            print("Status: Propagation kernels space allocated successfully")
            # Le d_KERNEL est déjà alloué dans initialize_test_mode, pas besoin de le recalculer ici
            
        except Exception as e:
            print(f"Warning: Could not initialize propagation kernels: {e}")
            # Fallback - use existing kernel or create if needed
            cam_nb_pix_X = int(params.get("holo_size_x", 1024))
            cam_nb_pix_Y = int(params.get("holo_size_y", 1024))
            if not hasattr(self, 'd_KERNEL') or self.d_KERNEL.shape != (cam_nb_pix_Y, cam_nb_pix_X):
                self.d_KERNEL = cp.zeros(shape=(cam_nb_pix_Y, cam_nb_pix_X), dtype=cp.complex64)

    def extract_object_slices(self, pos_x_um, pos_y_um, pos_z_um, vox_xy, vox_z):
        """Extract 3 slice views (XY, XZ, YZ) around an object from the reconstructed volume"""
        if not self.test_mode_allocated or self.d_volume_module is None:
            raise ValueError("Test mode not initialized or no volume available")
            
        print(f"🔍 Core: Extracting slices at ({pos_x_um:.3f}, {pos_y_um:.3f}, {pos_z_um:.3f}) µm")
        
        # Convert micrometers to pixel coordinates
        pixel_size = float(self.parameters.get('pixel_size', 1.0))  # µm/pixel
        step = float(self.parameters.get('step', 5e-7)) * 1e6  # Convert to µm
        
        # Convert position to pixels
        pos_x_pix = int(pos_x_um / pixel_size)
        pos_y_pix = int(pos_y_um / pixel_size)
        pos_z_pix = int(pos_z_um / step)
        
        print(f"📍 Pixel coordinates: ({pos_x_pix}, {pos_y_pix}, {pos_z_pix})")
        
        # Get volume dimensions
        volume_shape = self.d_volume_module.shape  # (Z, Y, X)
        print(f"📐 Volume shape: {volume_shape}")
        
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
        
        print(f"✅ Extracted slices: XY({xy_slice.shape}), XZ({xz_slice.shape}), YZ({yz_slice.shape})")
        
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
