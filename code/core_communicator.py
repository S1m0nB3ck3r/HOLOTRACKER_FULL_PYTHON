#!/usr/bin/env python3
"""
Communication system between Controller and Core using FIFO queues
Refactored with unified functions: NO DUPLICATION
"""
import os
import queue
import threading
import time
from enum import Enum
from dataclasses import dataclass
from typing import Any, Dict, Optional, Callable

class CommandType(Enum):
    """Types de commandes pour le Core"""
    ALLOCATE = "allocate"
    LOAD_MEAN_HOLO = "load_mean_holo"
    ENTER_TEST_MODE = "enter_test_mode"
    PROCESS_HOLOGRAM = "process_hologram" 
    PROCESS_HOLOGRAM_BATCH = "process_hologram_batch"
    CHANGE_PARAMETER = "change_parameter"
    EXIT_TEST_MODE = "exit_test_mode"
    ENTER_BATCH_MODE = "enter_batch_mode"
    EXIT_BATCH_MODE = "exit_batch_mode"
    EXTRACT_OBJECT_SLICES = "extract_object_slices"
    SHUTDOWN = "shutdown"

@dataclass
class Command:
    """Commande à exécuter par le Core"""
    type: CommandType
    data: Dict[str, Any]
    callback_id: Optional[str] = None

@dataclass
class Result:
    """Résultat retourné par le Core"""
    command_type: CommandType
    success: bool
    data: Dict[str, Any]
    error: Optional[str] = None
    callback_id: Optional[str] = None
    processing_times: Optional[Dict[str, float]] = None

class CoreCommunicator:
    """Gestionnaire de communication avec le Core sur thread séparé"""
    
    def __init__(self, core_instance):
        self.core = core_instance
        self.command_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.core_thread = None
        self.running = False
        self.callbacks = {}
        
        # Variables d'état
        self.current_state = "IDLE"
        self.allocation_needed = False
    
    def start(self):
        """Démarre le thread de communication avec le Core"""
        if not self.running:
            self.running = True
            self.core_thread = threading.Thread(target=self._core_worker, daemon=True)
            self.core_thread.start()
            # print("✅ CoreCommunicator thread started")
    
    def stop(self):
        """Arrête le thread de communication avec le Core"""
        if self.running:
            self.running = False
            # Envoyer commande de shutdown au Core
            self.send_command(CommandType.SHUTDOWN, {})
            if self.core_thread:
                self.core_thread.join(timeout=5.0)
            # print("✅ CoreCommunicator thread stopped")
    
    def send_command(self, command_type: CommandType, data: Dict[str, Any],
                    callback: Optional[Callable] = None) -> Optional[str]:
        """Envoie une commande au Core et retourne l'ID de callback"""
        callback_id = None
        if callback:
            import uuid
            callback_id = str(uuid.uuid4())
            self.callbacks[callback_id] = callback
        
        command = Command(type=command_type, data=data, callback_id=callback_id)
        self.command_queue.put(command)
        # print(f"📤 Command sent: {command_type.value}")
        return callback_id
    
    def get_result(self, timeout: float = 0.1) -> Optional[Result]:
        """Récupère un résultat du Core (non-bloquant)"""
        try:
            result = self.result_queue.get(timeout=timeout)
            
            # Traiter les callbacks
            if result.callback_id and result.callback_id in self.callbacks:
                callback = self.callbacks.pop(result.callback_id)
                try:
                    callback(result)
                except Exception as e:
                    # print(f"❌ Callback error: {e}")
                    pass
            
            return result
        except queue.Empty:
            return None
    
    def clear_command_queue(self):
        """Vide la queue des commandes (pour STOP BATCH)"""
        cleared_count = 0
        while not self.command_queue.empty():
            try:
                self.command_queue.get_nowait()
                cleared_count += 1
            except queue.Empty:
                break
        # print(f"🗑️ Cleared {cleared_count} commands from queue")
        return cleared_count
    
    def _core_worker(self):
        """Thread worker qui traite les commandes"""
        # print("🚀 Core worker thread started")
        
        while self.running:
            try:
                # Attendre une commande
                command = self.command_queue.get(timeout=1.0)
                
                # Traiter la commande
                result = self._process_command(command)
                
                # Envoyer le résultat
                self.result_queue.put(result)
                
            except queue.Empty:
                continue
            except Exception as e:
                # print(f"❌ Core worker error: {e}")
                # Créer un résultat d'erreur générique
                error_result = Result(
                    command_type=CommandType.SHUTDOWN,
                    success=False,
                    data={},
                    error=str(e)
                )
                self.result_queue.put(error_result)
        
        # print("🔄 Core worker thread stopping")
    
    def _process_command(self, command: Command) -> Result:
        """Traite une commande et retourne le résultat"""
        try:
            if command.type == CommandType.ALLOCATE:
                return self._handle_allocate(command)
                
            elif command.type == CommandType.LOAD_MEAN_HOLO:
                return self._handle_load_mean_holo(command)
                
            elif command.type == CommandType.ENTER_TEST_MODE:
                return self._handle_enter_test_mode(command)
                
            elif command.type == CommandType.PROCESS_HOLOGRAM:
                return self._handle_process_hologram(command)
                
            elif command.type == CommandType.PROCESS_HOLOGRAM_BATCH:
                return self._handle_process_hologram_batch(command)
                
            elif command.type == CommandType.CHANGE_PARAMETER:
                return self._handle_change_parameter(command)
                
            elif command.type == CommandType.EXIT_TEST_MODE:
                return self._handle_exit_test_mode(command)
                
            elif command.type == CommandType.ENTER_BATCH_MODE:
                return self._handle_enter_batch_mode(command)
                
            elif command.type == CommandType.EXIT_BATCH_MODE:
                return self._handle_exit_batch_mode(command)
                
            elif command.type == CommandType.EXTRACT_OBJECT_SLICES:
                return self._handle_extract_object_slices(command)
                
            elif command.type == CommandType.SHUTDOWN:
                self.running = False
                return Result(CommandType.SHUTDOWN, True, {})
                
            else:
                return Result(command.type, False, {}, f"Unknown command: {command.type}")
                
        except Exception as e:
            return Result(command.type, False, {}, f"Core command error ({command.type.value}): {str(e)}")
    
    # ==================== FONCTIONS UNIQUES (pas de duplication) ====================
    
    def _allocate_resources(self, params: dict) -> tuple[float, str]:
        """Fonction unique d'allocation GPU/mémoire"""
        allocation_start = time.perf_counter()
        
        # Set parameters in core and allocate
        self.core.set_parameters(**params)
        result_msg = self.core.allocate()
        
        allocation_time = time.perf_counter() - allocation_start
        # print(f"✅ Allocation completed in {allocation_time:.3f}s")
        
        return allocation_time, result_msg
    
    def _process_hologram_pipeline(self, directory: str, filename: str) -> dict:
        """Fonction unique de traitement d'hologramme"""
        # print(f"🚀 Processing hologram: {filename}")
        
        # Pipeline complet de traitement
        self.core.process_hologram_complete_pipeline(directory, filename)
        
        # Récupération des résultats
        results_data = self.core.get_3d_results_data()
        
        # Ajout des features brutes pour le mode batch
        if hasattr(self.core, 'results') and self.core.results and 'features' in self.core.results:
            results_data['features'] = self.core.results['features']
        
        # Ajout des infos directory/filename
        if results_data:
            results_data['directory'] = directory
            results_data['filename'] = filename
        
        return results_data
    
    def _create_result(self, command_type: CommandType, results_data: dict, allocation_time: float = None) -> Result:
        """Fonction unique de création de résultat"""
        if allocation_time and results_data and 'processing_times' in results_data:
            results_data['processing_times']['allocation'] = allocation_time
        
        processing_times = results_data.get('processing_times') if results_data else None
        if allocation_time and not processing_times:
            processing_times = {'allocation': allocation_time}
        
        return Result(
            command_type=command_type,
            success=True,
            data=results_data,
            processing_times=processing_times
        )
    
    # ==================== HANDLERS UTILISANT LES FONCTIONS UNIQUES ====================
    
    def _handle_allocate(self, command: Command) -> Result:
        """ALLOCATE: Allocation GPU/mémoire (partagée TEST/BATCH)"""
        try:
            params = command.data.get('parameters', {})
            allocation_time, result_msg = self._allocate_resources(params)
            
            return self._create_result(CommandType.ALLOCATE, {
                'message': result_msg
            }, allocation_time)
            
        except Exception as e:
            return Result(CommandType.ALLOCATE, False, {}, f"Allocation error: {str(e)}")
    
    def _handle_load_mean_holo(self, command: Command) -> Result:
        """LOAD_MEAN_HOLO: Chargement de l'hologramme moyen"""
        try:
            params = command.data.get('parameters', {})
            load_start = time.perf_counter()
            
            # Set parameters and load mean hologram
            self.core.set_parameters(**params)
            self.core.load_mean_hologram()
            
            load_time = time.perf_counter() - load_start
            # print(f"✅ Mean hologram loaded in {load_time:.3f}s")
            
            return self._create_result(CommandType.LOAD_MEAN_HOLO, {
                'message': 'Mean hologram loaded successfully'
            }, load_time)
            
        except Exception as e:
            return Result(CommandType.LOAD_MEAN_HOLO, False, {}, f"Load mean hologram error: {str(e)}")
    
    def _handle_enter_test_mode(self, command: Command) -> Result:
        """ENTER_TEST_MODE: Changement d'état seulement (plus d'allocation)"""
        try:
            self.current_state = "TEST_MODE"
            # Call core method to set mode
            self.core.enter_test_mode()
            
            return self._create_result(CommandType.ENTER_TEST_MODE, {
                'message': 'Test mode activated'
            }, 0.0)
            
        except Exception as e:
            return Result(CommandType.ENTER_TEST_MODE, False, {}, f"Enter test mode error: {str(e)}")
    
    def _handle_process_hologram(self, command: Command) -> Result:
        """PROCESS_HOLOGRAM: Traitement + Affichage"""
        try:
            directory = command.data['directory']
            filename = command.data['filename']
            
            # Traitement (pas d'allocation, déjà faite)
            results_data = self._process_hologram_pipeline(directory, filename)
            
            return self._create_result(CommandType.PROCESS_HOLOGRAM, results_data)
            
        except Exception as e:
            # print(f"❌ Core error processing hologram: {e}")
            return Result(CommandType.PROCESS_HOLOGRAM, False, {}, f"Process hologram error: {str(e)}")
    
    def _handle_process_hologram_batch(self, command: Command) -> Result:
        """PROCESS_HOLOGRAM_BATCH: Traitement + Écriture CSV (pas d'affichage)"""
        try:
            directory = command.data['directory']
            filename = command.data['filename']
            
            # Traitement (réutilise la fonction unifiée)
            results_data = self._process_hologram_pipeline(directory, filename)
            
            # Écriture dans le CSV (mode batch)
            if hasattr(self, 'batch_csv_path') and results_data:
                self.batch_hologram_counter += 1
                
                # Extraire les données depuis les features (format Core)
                features = results_data.get('features')
                if features is not None and len(features) > 0:
                    # Prendre le premier objet détecté (le plus significatif)
                    main_feature = features[0]
                    x = main_feature[1]  # baryX
                    y = main_feature[2]  # baryY  
                    z = main_feature[3]  # baryZ
                    nb_pix = int(main_feature[4])  # nb_pix
                    
                    # Écrire la ligne dans le CSV
                    with open(self.batch_csv_path, 'a', newline='', encoding='utf-8') as f:
                        f.write(f"{self.batch_hologram_counter},{x:.2f},{y:.2f},{z:.2f},{nb_pix},{filename}\n")
                    
                    csv_written = True
                    # print(f"📄 CSV written: #{self.batch_hologram_counter} - {filename} - {len(features)} objects")
                else:
                    # Pas d'objets détectés, écrire une ligne avec des valeurs par défaut
                    with open(self.batch_csv_path, 'a', newline='', encoding='utf-8') as f:
                        f.write(f"{self.batch_hologram_counter},0.00,0.00,0.00,0,{filename}\n")
                    
                    csv_written = True
                    # print(f"📄 CSV written: #{self.batch_hologram_counter} - {filename} - No objects detected")
                
                # Ajouter info batch au résultat
                results_data['batch_info'] = {
                    'hologram_number': self.batch_hologram_counter,
                    'csv_written': csv_written,
                    'csv_path': self.batch_csv_path
                }
            
            return self._create_result(CommandType.PROCESS_HOLOGRAM_BATCH, results_data)
            
        except Exception as e:
            # print(f"❌ Core error processing hologram batch: {e}")
            return Result(CommandType.PROCESS_HOLOGRAM_BATCH, False, {}, f"Process hologram batch error: {str(e)}")
    
    def _handle_change_parameter(self, command: Command) -> Result:
        """CHANGE_PARAMETER: Allocation conditionnelle + Traitement + Affichage"""
        try:
            param_name = command.data['name']
            param_value = command.data['value']
            directory = command.data.get('directory', '')
            filename = command.data.get('filename', '')
            
            # Paramètres nécessitant une réallocation
            reallocation_params = ['holo_size_x', 'holo_size_y', 'number_of_planes']
            
            # Paramètres nécessitant d'invalider le cache de l'hologramme moyen
            mean_holo_cache_params = ['mean_hologram_image_path']
            
            # Paramètres nécessitant d'invalider le cache du seuil
            threshold_cache_params = ['cleaning_type', 'remove_mean', 'high_pass', 'low_pass', 'focus_type', 'sum_size']
            
            # Mettre à jour le paramètre using set_parameters
            self.core.set_parameters(**{param_name: param_value})
            
            # Invalider le cache de l'hologramme moyen si nécessaire
            if param_name in mean_holo_cache_params:
                self.core.h_mean_holo = None
                print(f"🔄 Parameter {param_name} changed, invalidating mean hologram cache")
            
            # Invalider le cache du seuil si nécessaire
            if param_name in threshold_cache_params:
                self.core.threshold = None
                self.core.last_nb_StdVar_threshold = None
                print(f"🔄 Parameter {param_name} changed, invalidating threshold cache")
            
            allocation_time = None
            
            # Allocation conditionnelle
            if param_name in reallocation_params:
                # print(f"🔄 Parameter {param_name} = {param_value} requires reallocation")
                # Get current parameters as dict for reallocation
                current_params = self.core.get_parameters_dict()
                allocation_time, _ = self._allocate_resources(current_params)
            # else:
            #     print(f"📝 Parameter {param_name} = {param_value} updated (no reallocation)")
            
            # Traitement (si hologramme disponible)
            if directory and filename:
                results_data = self._process_hologram_pipeline(directory, filename)
                if results_data and param_name in reallocation_params:
                    results_data['reallocation'] = True
                return self._create_result(CommandType.CHANGE_PARAMETER, results_data, allocation_time)
            else:
                # Pas d'hologramme, juste confirmer le changement
                data = {
                    'message': f'Parameter {param_name} updated', 
                    'reallocation': param_name in reallocation_params
                }
                return self._create_result(CommandType.CHANGE_PARAMETER, data, allocation_time)
                
        except Exception as e:
            # print(f"❌ Core error in parameter change: {e}")
            return Result(CommandType.CHANGE_PARAMETER, False, {}, f"Parameter change error: {str(e)}")
    
    def _handle_exit_test_mode(self, command: Command) -> Result:
        """Gère la sortie du mode test"""
        try:
            self.current_state = "IDLE"
            # Call core method to exit test mode
            self.core.exit_test_mode()
            
            return Result(
                command_type=CommandType.EXIT_TEST_MODE,
                success=True,
                data={'message': 'Test mode exited'}
            )
            
        except Exception as e:
            return Result(CommandType.EXIT_TEST_MODE, False, {}, f"Exit test mode error: {str(e)}")
    
    def _handle_enter_batch_mode(self, command: Command) -> Result:
        """ENTER_BATCH_MODE: Changement d'état + Initialisation CSV (plus d'allocation)"""
        try:
            directory = command.data.get('directory', '')
            
            self.current_state = "BATCH_MODE"
            # Call core method to enter batch mode
            self.core.enter_batch_mode()
            
            # Initialisation du fichier CSV SANS timestamp
            if directory:
                csv_filename = "RESULT.csv"
                csv_path = os.path.join(directory, csv_filename)
                
                # Créer le fichier CSV avec en-tête
                with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                    f.write("N_Hologramme,barycentre_X,barycentre_Y,barycentre_Z,nb_pix,filename\n")
                
                # Stocker le path du CSV pour les écritures suivantes
                self.batch_csv_path = csv_path
                self.batch_hologram_counter = 0
                
                return self._create_result(CommandType.ENTER_BATCH_MODE, {
                    'message': f'Batch mode activated, CSV created: {csv_filename}',
                    'csv_path': csv_path
                }, 0.0)
            else:
                return Result(CommandType.ENTER_BATCH_MODE, False, {}, "No directory specified for batch processing")
            
        except Exception as e:
            return Result(CommandType.ENTER_BATCH_MODE, False, {}, f"Enter batch mode error: {str(e)}")
    
    def _handle_exit_batch_mode(self, command: Command) -> Result:
        """EXIT_BATCH_MODE: Nettoyage et fermeture"""
        try:
            self.current_state = "IDLE"
            # Call core method to exit batch mode
            self.core.exit_batch_mode()
            
            # Nettoyage des variables batch
            if hasattr(self, 'batch_csv_path'):
                delattr(self, 'batch_csv_path')
            if hasattr(self, 'batch_hologram_counter'):
                delattr(self, 'batch_hologram_counter')
            
            return Result(
                command_type=CommandType.EXIT_BATCH_MODE,
                success=True,
                data={'message': 'Batch mode exited'}
            )
            
        except Exception as e:
            return Result(CommandType.EXIT_BATCH_MODE, False, {}, f"Exit batch mode error: {str(e)}")

    def _handle_extract_object_slices(self, command: Command) -> Result:
        """Extract 3 slice views around an object from the reconstructed volume"""
        try:
            data = command.data
            object_num = data['object_num']
            position = data['position']  # [x, y, z] in micrometers
            vox_xy = data['vox_xy']
            vox_z = data['vox_z']
            display_in_dialog = data.get('display_in_dialog', False)
            
            # Get slices from core
            slices_data = self.core.extract_object_slices(position[0], position[1], position[2], vox_xy, vox_z)
            
            return Result(
                command_type=CommandType.EXTRACT_OBJECT_SLICES,
                success=True,
                data={
                    'object_num': object_num,
                    'position': position,
                    'slices': slices_data,
                    'display_in_dialog': display_in_dialog
                }
            )
            
        except Exception as e:
            return Result(CommandType.EXTRACT_OBJECT_SLICES, False, {}, str(e))
