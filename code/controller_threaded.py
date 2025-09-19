import threading
import queue
import os
import numpy as np
import tkinter as tk
from PIL import Image
try:
    import cupy as cp
except ImportError:
    cp = None
from core_communicator import CoreCommunicator, CommandType

class HoloTrackerController:
    def __init__(self, app_ui, core):
        self.ui = app_ui
        self.core = core
        self.queue = queue.Queue()
        self.running = False
        self.state = "WAIT"
        
        # Variables pour BATCH processing
        self.last_batch_result = None
        self.batch_total_files = 0
        self.batch_iteration_start_time = None
        self.last_iteration_time = None
        
        # Nouvelle architecture threadée
        self.core_comm = CoreCommunicator(core)
        self.core_comm.start()
        
        # Timer pour vérifier les résultats
        self.ui.root.after(100, self.check_core_results)
        
        self.ui.update_buttons_state(self.state)

    def update_status(self, text):
        self.ui.status_var.set(f"[{self.state}] {text}")

    def set_state(self, new_state):
        self.state = new_state
        self.ui.update_buttons_state(self.state)
    
    def check_core_results(self):
        """Vérifie périodiquement les résultats du Core (appelé toutes les 100ms)"""
        try:
            result = self.core_comm.get_result()
            if result:
                self.handle_core_result(result)
        except Exception as e:
            print(f"❌ Error checking core results: {e}")
        finally:
            # Planifier la prochaine vérification
            if self.running or self.core_comm.running:
                self.ui.root.after(100, self.check_core_results)
    
    def handle_core_result(self, result):
        """Gère les résultats reçus du Core"""
        print(f"📥 Received result: {result.command_type.value} - Success: {result.success}")
        
        if not result.success:
            self.update_status(f"Error: {result.error}")
            return
        
        if result.command_type == CommandType.ENTER_TEST_MODE:
            self.set_state("TEST_MODE")
            
            # Auto-définir le display selon Remove mean Hologram
            remove_mean = self.ui.parameters.get('remove_mean', False)
            if remove_mean:
                self.ui.display_combobox.set("CLEANED_HOLOGRAM")
                print("🎯 Auto-display: CLEANED_HOLOGRAM (Remove mean = True)")
            else:
                self.ui.display_combobox.set("RAW_HOLOGRAM")
                print("🎯 Auto-display: RAW_HOLOGRAM (Remove mean = False)")
            
            # Traitement unifié: ENTER_TEST_MODE fait maintenant Allocation + Pipeline
            if result.data.get('cluster_positions') is not None:
                # On a des résultats complets (allocation + pipeline)
                self.update_status("Test mode activated - Complete processing done")
                try:
                    print("📊 Controller: Displaying complete results from ENTER_TEST_MODE")
                    self.ui.display_results(result.data)
                    print("✅ Controller: ENTER_TEST_MODE results displayed successfully")
                except Exception as e:
                    print(f"❌ Controller: Error displaying ENTER_TEST_MODE results: {e}")
            else:
                # Juste l'allocation (pas d'hologramme sélectionné)
                self.update_status("Test mode activated - Allocation completed, select hologram")
                
                # Afficher l'image selon le display type choisi si hologramme disponible
                directory = self.ui.dir_text.get("1.0", "end-1c").strip()
                filename = self.ui.hologram_combobox.get()
                if directory and filename:
                    try:
                        print("🖼️  Controller: Updating image display with new display type")
                        self.ui.on_display_changed()
                        print("✅ Controller: Image display updated successfully")
                    except Exception as e:
                        print(f"❌ Controller: Error updating image display: {e}")
            
        elif result.command_type == CommandType.PROCESS_HOLOGRAM:
            # Afficher les résultats et timing avec gestion d'erreur robuste
            try:
                print("📊 Controller: About to call ui.display_results")
                self.ui.display_results(result.data)
                print("✅ Controller: ui.display_results completed successfully")
            except Exception as e:
                print(f"❌ Controller: Error in ui.display_results: {e}")
                # Continuer même si l'affichage 3D échoue
            
            # Les processing_times sont déjà dans result.data, donc pas besoin d'appel séparé
            # mais on peut faire un appel explicite pour s'assurer que le timing s'affiche
            try:
                if result.processing_times:
                    timing_data = {'processing_times': result.processing_times}
                    self.ui.update_timing_display(timing_data)
                    print("✅ Controller: Timing display updated successfully")
            except Exception as e:
                print(f"❌ Controller: Error updating timing display: {e}")
            
            # Afficher l'image
            directory = result.data.get('directory', '')
            filename = result.data.get('filename', '')
            if directory and filename:
                try:
                    display_type = self.ui.display_combobox.get()
                    plane_number = int(self.ui.plane_number_spinbox.get()) if self.ui.plane_number_spinbox.get() else 0
                    result_image = self.core.get_display_image(directory, filename, display_type, plane_number)
                    self.ui.display_hologram_image(result_image)
                    print("✅ Controller: Image display updated successfully")
                except Exception as e:
                    print(f"❌ Controller: Error displaying image: {e}")
            
            num_objects = result.data.get('count', 0)
            self.update_status(f"Hologram processed - Found {num_objects} objects")
            
        elif result.command_type == CommandType.CHANGE_PARAMETER:
            if result.data.get('reallocation'):
                self.update_status("Parameter changed - Reallocation completed")
            else:
                self.update_status("Parameter updated")
            
            # Auto-retraiter l'hologramme courant après tout changement de paramètre
            directory = self.ui.dir_text.get("1.0", "end-1c").strip()
            filename = self.ui.hologram_combobox.get()
            if directory and filename:
                self.core_comm.send_command(
                    CommandType.PROCESS_HOLOGRAM,
                    {'directory': directory, 'filename': filename}
                )
                
        elif result.command_type == CommandType.EXIT_TEST_MODE:
            self.set_state("WAIT")
            self.update_status("Test mode exited")
        
        # ==================== BATCH MODE RESULT HANDLERS ====================
        elif result.command_type == CommandType.ALLOCATE:
            # Allocation pour TEST_MODE ou BATCH_MODE
            allocation_time = result.processing_times.get('allocation', 0) if result.processing_times else 0
            self.update_status(f"Allocation completed: {allocation_time:.3f}s")
            
        elif result.command_type == CommandType.ENTER_BATCH_MODE:
            self.set_state("BATCH_MODE")
            csv_path = result.data.get('csv_path', '')
            if csv_path:
                csv_filename = os.path.basename(csv_path)
                self.update_status(f"Batch mode activated - CSV: {csv_filename}")
                # Mise à jour UI: activer les contrôles batch, griser les paramètres
                self.ui.on_batch_mode_entered(csv_filename)
            else:
                self.update_status("Batch mode activated")
                
        elif result.command_type == CommandType.PROCESS_HOLOGRAM_BATCH:
            # Calculer le temps entre itérations (pipeline complet)
            import time
            current_time = time.perf_counter()
            if self.batch_iteration_start_time is not None:
                self.last_iteration_time = current_time - self.batch_iteration_start_time
                print(f"⏱️  CONTROLLER: Iteration time calculated: {self.last_iteration_time*1000:.1f}ms")
            else:
                print("⏱️  CONTROLLER: First iteration, no previous time")
            self.batch_iteration_start_time = current_time
            
            # Stocker les résultats du dernier hologramme traité
            self.last_batch_result = result.data
            
            # Essayer de trouver la checkbox DISPLAY RESULTS
            display_checked = False
            
            # Chercher spécifiquement la checkbox
            if hasattr(self.ui, 'display_results_checkbox'):
                # Essayer d'accéder à la variable via la checkbox
                try:
                    checkbox = self.ui.display_results_checkbox
                    if hasattr(checkbox, 'cget'):
                        var_name = checkbox.cget('variable')
                        if var_name:
                            display_checked = var_name.get()
                            print(f"✅ DEBUG: Found checkbox via cget: {display_checked}")
                        else:
                            print("❌ DEBUG: checkbox variable is None")
                    else:
                        print("❌ DEBUG: checkbox has no cget method")
                except Exception as e:
                    print(f"❌ DEBUG: Error accessing checkbox: {e}")
            
            # Si pas trouvé, essayer les noms de variables directement
            if not display_checked:
                for attr_name in dir(self.ui):
                    if 'display' in attr_name.lower() and 'var' in attr_name.lower():
                        try:
                            attr_obj = getattr(self.ui, attr_name)
                            if hasattr(attr_obj, 'get'):
                                value = attr_obj.get()
                                print(f"🔍 DEBUG: Found variable {attr_name} = {value}")
                                if attr_name in ['display_results_var', 'display_batch_results_var']:
                                    display_checked = value
                                    print(f"✅ DEBUG: Using {attr_name} = {display_checked}")
                                    break
                        except:
                            pass
            
            print(f"🔍 DEBUG: Final display_checked = {display_checked}")
            
            # TOUJOURS mettre à jour les temps d'exécution en mode BATCH
            if 'processing_times' in result.data and self.last_iteration_time:
                # Créer une copie des processing_times avec iteration_time
                timing_with_iteration = result.data['processing_times'].copy()
                timing_with_iteration['iteration_time'] = self.last_iteration_time
                print(f"⏱️  BATCH: Calling update_timing_display with iteration_time: {self.last_iteration_time*1000:.1f}ms")
                self.ui.update_timing_display({'processing_times': timing_with_iteration})
            
            # Afficher si checkbox DISPLAY RESULTS est cochée
            if display_checked:
                print("🖼️  BATCH: Calling _display_batch_results()")
                self._display_batch_results(result.data)
            else:
                print("❌ BATCH: Display not enabled or checkbox not found")
            
            # Afficher directement le temps d'itération dans le status
            if self.last_iteration_time:
                iteration_ms = self.last_iteration_time * 1000
                processing_ms = result.data.get('processing_times', {}).get('total_processing', 0) * 1000
                print(f"⏱️  BATCH: Processing: {processing_ms:.0f}ms, Iteration: {iteration_ms:.1f}ms")
                
                # Afficher les timing avec iteration dans le status
                batch_info = result.data.get('batch_info', {})
                if batch_info:
                    hologram_number = batch_info.get('hologram_number', 0)
                    self.update_status(f"Batch {hologram_number}/{self.batch_total_files} - Processing: {processing_ms:.0f}ms, Iteration: {iteration_ms:.1f}ms")
            
            # Progression batch
            batch_info = result.data.get('batch_info', {})
            if batch_info:
                hologram_number = batch_info.get('hologram_number', 0)
                total_files = getattr(self, 'batch_total_files', '?')
                self.update_status(f"Batch processing: {hologram_number}/{total_files}")
            
            num_objects = result.data.get('count', 0)
            filename = result.data.get('filename', '')
            print(f"✅ Batch hologram {filename} processed - Found {num_objects} objects")
            
        elif result.command_type == CommandType.EXIT_BATCH_MODE:
            self.set_state("WAIT")
            self.update_status("Batch mode exited")
            # Mise à jour UI: réactiver les contrôles normaux
            self.ui.on_batch_mode_exited()
            
        elif result.command_type == CommandType.EXTRACT_OBJECT_SLICES:
            if result.success:
                object_num = result.data['object_num']
                slices_data = result.data['slices']
                print(f"✅ Slices extracted for object #{object_num}")
                # Send to UI for display
                self.ui.display_object_slices(object_num, slices_data)
            else:
                print(f"❌ Failed to extract slices: {result.error}")
                self.update_status(f"Error extracting slices: {result.error}")

    def on_parameter_changed(self, name, value):
        try:
            if self.state == "TEST_MODE":
                # Envoyer commande de changement de paramètre au Core
                self.core_comm.send_command(
                    CommandType.CHANGE_PARAMETER,
                    {'name': name, 'value': value}
                )
            else:
                # En mode WAIT, mise à jour directe
                result = self.core.set_parameter(name, value)
                self.update_status(f"{name} : {value}")
                
        except Exception as e:
            self.update_status(f"Error: {str(e)}")
    
    def sync_parameters_to_core(self):
        """Synchronize parameters one by one with explicit validation"""
        if not hasattr(self.ui, 'parameters') or not self.ui.parameters:
            return
            
        params = self.ui.parameters
        synced_count = 0
        
        try:
            # === HOLOGRAM SETUP ===
            if "mean_hologram_image_path" in params:
                self.core.set_parameter("mean_hologram_image_path", str(params["mean_hologram_image_path"]))
                synced_count += 1
                
            if "holograms_directory" in params:
                self.core.set_parameter("holograms_directory", str(params["holograms_directory"]))
                synced_count += 1
                
            if "image_type" in params:
                self.core.set_parameter("image_type", str(params["image_type"]))
                synced_count += 1
            
            # === OPTICAL PARAMETERS ===
            if "wavelength" in params:
                self.core.set_parameter("medium_wavelength", float(params["wavelength"]))
                synced_count += 1
                
            if "optical_index" in params:
                self.core.set_parameter("optical_index", float(params["optical_index"]))
                synced_count += 1
                
            if "magnification" in params:
                self.core.set_parameter("magnification", float(params["magnification"]))
                synced_count += 1
                
            if "pixel_size_camera" in params:
                self.core.set_parameter("pixel_size_camera", float(params["pixel_size_camera"]))
                synced_count += 1
            
            # === RECONSTRUCTION PARAMETERS ===
            if "holo_size_x" in params:
                self.core.set_parameter("holo_size_x", int(params["holo_size_x"]))
                synced_count += 1
                
            if "holo_size_y" in params:
                self.core.set_parameter("holo_size_y", int(params["holo_size_y"]))
                synced_count += 1
                
            if "z_start" in params:
                self.core.set_parameter("z_start", float(params["z_start"]))
                synced_count += 1
                
            if "z_end" in params:
                self.core.set_parameter("z_end", float(params["z_end"]))
                synced_count += 1
                
            if "dz" in params:
                self.core.set_parameter("dz", float(params["dz"]))
                synced_count += 1
                
            if "nb_plane" in params:
                self.core.set_parameter("nb_plane", int(params["nb_plane"]))
                synced_count += 1
            
            # === FILTERING PARAMETERS ===
            if "remove_mean" in params:
                self.core.set_parameter("remove_mean", bool(params["remove_mean"]))
                synced_count += 1
                
            if "filter_activated" in params:
                self.core.set_parameter("filter_activated", bool(params["filter_activated"]))
                synced_count += 1
                
            if "filter_size_freq" in params:
                self.core.set_parameter("filter_size_freq", float(params["filter_size_freq"]))
                synced_count += 1
                
            if "filter_shape" in params:
                self.core.set_parameter("filter_shape", str(params["filter_shape"]))
                synced_count += 1
            
            # === LOCALIZATION PARAMETERS ===
            if "threshold_method" in params:
                self.core.set_parameter("threshold_method", str(params["threshold_method"]))
                synced_count += 1
                
            if "threshold_value" in params:
                self.core.set_parameter("threshold_value", float(params["threshold_value"]))
                synced_count += 1
                
            if "min_area" in params:
                self.core.set_parameter("min_area", int(params["min_area"]))
                synced_count += 1
                
            if "max_area" in params:
                self.core.set_parameter("max_area", int(params["max_area"]))
                synced_count += 1
            
            self.update_status(f"Parameters synchronized ({synced_count} params)")
            
        except Exception as e:
            self.update_status(f"Error syncing parameters: {str(e)}")

    def on_enter_test_mode(self):
        if self.state == "WAIT":
            self.update_status("Entering test mode...")
            
            # Préparer les paramètres depuis l'UI
            self.ui.update_parameters_from_ui()
            
            # Récupérer hologramme sélectionné
            directory = self.ui.dir_text.get("1.0", "end-1c").strip()
            filename = self.ui.hologram_combobox.get()
            
            # Nouveau workflow séparé: ENTER_TEST_MODE + ALLOCATE + optionnel PROCESS
            print("🚀 Controller: ENTER_TEST_MODE - separate allocation")
            
            # 1. Changement d'état
            self.core_comm.send_command(CommandType.ENTER_TEST_MODE, {})
            
            # 2. Allocation
            self.core_comm.send_command(CommandType.ALLOCATE, {'parameters': self.ui.parameters})
            
            # 3. Traitement (si hologramme disponible)
            if directory and filename:
                self.core_comm.send_command(CommandType.PROCESS_HOLOGRAM, {
                    'directory': directory, 
                    'filename': filename
                })
                print(f"🚀 Controller: Processing {filename} after allocation")

    def on_exit_test_mode(self):
        if self.state == "TEST_MODE":
            self.update_status("Exiting test mode...")
            
            # Envoyer commande de sortie du mode test au Core
            self.core_comm.send_command(
                CommandType.EXIT_TEST_MODE,
                {}
            )

    def on_hologram_selected(self, directory, filename):
        if self.state == "TEST_MODE":
            self.update_status(f"Processing {filename}...")
            
            # Envoyer commande de traitement d'hologramme au Core
            self.core_comm.send_command(
                CommandType.PROCESS_HOLOGRAM,
                {
                    'directory': directory,
                    'filename': filename
                }
            )
        else:
            self.update_status("Enter test mode to process holograms.")

    def on_display_changed(self, display_type, plane_number):
        """Handle display type or plane number change - update image immediately"""
        if self.state == "TEST_MODE":
            filename = self.ui.hologram_combobox.get()
            directory = self.ui.dir_text.get("1.0", tk.END).strip()
            if filename and directory:
                try:
                    # Update image display immediately
                    result_image = self.core.get_display_image(directory, filename, display_type, plane_number)
                    self.ui.display_hologram_image(result_image)
                    self.update_status(f"Display updated to {display_type} - Plane {plane_number}")
                except Exception as e:
                    self.update_status(f"Error updating display: {str(e)}")

    # ==================== BATCH MODE METHODS ====================
    
    def on_enter_batch_mode(self, batch_directory):
        """Start batch processing mode with FIFO approach"""
        if self.state == "WAIT":
            self.update_status("Preparing batch processing...")
            
            # Préparer les paramètres depuis l'UI
            self.ui.update_parameters_from_ui()
            
            # Lister les fichiers selon image_type
            image_type = self.ui.parameters.get('image_type', 'bmp')
            try:
                files = [f for f in os.listdir(batch_directory) 
                        if f.lower().endswith(f'.{image_type.lower()}')]
                files.sort()
            except Exception as e:
                self.update_status(f"Error reading directory: {str(e)}")
                return
            
            if not files:
                self.update_status(f"No {image_type} files found in directory")
                return
            
            display_results = getattr(self.ui, 'display_results_var', None)
            display_enabled = display_results.get() if display_results else False
            
            # Tracker le nombre total pour progression
            self.batch_total_files = len(files)
            
            # Réinitialiser le timing des itérations
            self.batch_iteration_start_time = None
            self.last_iteration_time = None
            
            # Auto-définir DISPLAY selon Remove mean Hologram (toujours au démarrage BATCH)
            remove_mean = self.ui.parameters.get('remove_mean', False)
            if remove_mean:
                self.ui.display_combobox.set("CLEANED_HOLOGRAM")
                print("🎯 BATCH Auto-display: CLEANED_HOLOGRAM (Remove mean = True)")
            else:
                self.ui.display_combobox.set("RAW_HOLOGRAM")
                print("🎯 BATCH Auto-display: RAW_HOLOGRAM (Remove mean = False)")
        
        # Envoyer TOUTES les commandes dans la FIFO
        self._send_batch_commands_to_fifo(batch_directory, files, display_enabled)
        
        self.update_status(f"Batch started: {len(files)} files queued")

    def _send_batch_commands_to_fifo(self, directory, files, display_results):
        """Send all batch commands to FIFO at once"""
        print(f"🚀 Controller: Sending batch commands for {len(files)} files")
        
        # 1. Création CSV + état BATCH
        self.core_comm.send_command(CommandType.ENTER_BATCH_MODE, {'directory': directory})
        
        # 2. Allocation GPU
        self.core_comm.send_command(CommandType.ALLOCATE, {'parameters': self.ui.parameters})
        
        # 3. Traitement de chaque fichier (plus de display_results dans FIFO)
        for filename in files:
            command_data = {
                'directory': directory, 
                'filename': filename
            }
            self.core_comm.send_command(CommandType.PROCESS_HOLOGRAM_BATCH, command_data)
        
        print(f"📤 Controller: {len(files) + 2} commands sent to FIFO (ENTER_BATCH + ALLOCATE + {len(files)} PROCESS)")

    def _display_batch_results(self, result_data):
        """Affiche les résultats batch selon le DISPLAY sélectionné dynamiquement"""
        try:
            print("🖼️  DEBUG: Starting batch results display...")
            
            # Afficher les résultats (graphique 3D, timing, etc.)
            print("🖼️  DEBUG: Calling ui.display_results()")
            self.ui.display_results(result_data)
            print("✅ Controller: Batch results displayed (3D plot, timing)")
            
            # Afficher l'image selon le DISPLAY actuel (peut changer pendant le batch)
            display_type = self.ui.display_combobox.get()
            plane_number_str = self.ui.plane_number_spinbox.get()
            plane_number = int(plane_number_str) if plane_number_str else 0
            
            print(f"🖼️  DEBUG: display_type={display_type}, plane_number={plane_number}")
            
            directory = result_data.get('directory', '')
            filename = result_data.get('filename', '')
            print(f"🖼️  DEBUG: directory={directory}, filename={filename}")
            
            if directory and filename:
                print(f"🖼️  DEBUG: Calling core.get_display_image({display_type}, {plane_number})")
                result_image = self.core.get_display_image(directory, filename, display_type, plane_number)
                
                if result_image is not None:
                    print("🖼️  DEBUG: Calling ui.display_hologram_image()")
                    self.ui.display_hologram_image(result_image)
                    print(f"✅ Controller: Batch image displayed ({display_type}, plane {plane_number})")
                else:
                    print("❌ Controller: get_display_image returned None")
            else:
                print("❌ Controller: Missing directory or filename")
                
        except Exception as e:
            print(f"❌ Controller: Error displaying batch results: {e}")
            import traceback
            traceback.print_exc()

    def on_exit_batch_mode(self):
        """Exit batch processing mode and clear FIFO"""
        if self.state == "BATCH_MODE":
            # VIDER LA FIFO
            cleared = self.core_comm.clear_command_queue()
            self.update_status(f"Batch stopped - {cleared} commands cancelled")
            
            # Envoyer commande de sortie du mode batch au Core
            self.core_comm.send_command(CommandType.EXIT_BATCH_MODE, {})

    def on_process_hologram_batch(self, directory, filename):
        """Process a single hologram in batch mode"""
        if self.state == "BATCH_MODE":
            self.update_status(f"Batch processing {filename}...")
            
            # Envoyer commande de traitement d'hologramme batch au Core
            self.core_comm.send_command(
                CommandType.PROCESS_HOLOGRAM_BATCH,
                {
                    'directory': directory,
                    'filename': filename
                }
            )
        else:
            self.update_status("Enter batch mode to process holograms in batch.")

    def on_holograms_directory_changed(self, directory):
        """Handle change in holograms directory"""
        try:
            self.core.set_parameter("holograms_directory", directory)
            
            # Update the hologram list in UI
            if os.path.exists(directory):
                supported_extensions = ['.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp']
                files = [f for f in os.listdir(directory) 
                        if any(f.lower().endswith(ext) for ext in supported_extensions)]
                files.sort()
                
                # Update UI combobox
                self.ui.hologram_combobox['values'] = files
                if files:
                    self.ui.hologram_combobox.set(files[0])
                    self.update_status(f"Directory loaded: {len(files)} holograms found")
                else:
                    self.update_status("Directory loaded: No supported images found")
            else:
                self.update_status("Invalid directory path")
                
        except Exception as e:
            self.update_status(f"Error loading directory: {str(e)}")

    def handle_parameter_changed(self, data):
        """Handle parameter change event from message manager"""
        if isinstance(data, dict) and 'name' in data and 'value' in data:
            self.on_parameter_changed(data['name'], data['value'])

    def on_batch_process(self, data):
        """Handle batch processing request - Legacy method"""
        # Redirect to new batch system
        directory = self.ui.dir_text.get("1.0", "end-1c").strip()
        if directory:
            self.on_enter_batch_mode(directory)
        else:
            self.update_status("Please select a directory for batch processing")

    def on_cancel_batch(self):
        """Handle cancel batch processing - Legacy method"""
        # Redirect to new batch system
        self.on_exit_batch_mode()

    def extract_object_slices(self, object_num, pos_x, pos_y, pos_z, vox_xy, vox_z):
        """Extract 3 slice views around an object from the reconstructed volume"""
        if self.state != "TEST":
            self.update_status("Object viewer only available in TEST mode")
            return
            
        # Send command to core to extract slices
        command_data = {
            'object_num': object_num,
            'position': [pos_x, pos_y, pos_z],
            'vox_xy': vox_xy,
            'vox_z': vox_z
        }
        
        print(f"🔍 Controller: Requesting slices for object #{object_num} at ({pos_x:.3f}, {pos_y:.3f}, {pos_z:.3f})")
        print(f"📐 Slice sizes: XY={vox_xy}x{vox_xy}, Z={vox_z}")
        
        self.core_comm.send_command(CommandType.EXTRACT_OBJECT_SLICES, command_data)

    def cleanup(self):
        """Cleanup resources"""
        if self.core_comm:
            self.core_comm.stop()
        if self.running:
            self.running = False
