import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import json
import os

class ScientificFloatEntry(ttk.Entry):
    def __init__(self, master=None, **kwargs):
        ttk.Entry.__init__(self, master, **kwargs)
        self.valid = (self.register(self._validate_scientific_float), '%P')
        self.config(validate='key', validatecommand=self.valid)
    def _validate_scientific_float(self, new_value):
        if new_value == "":
            return True
        try:
            float(new_value)
            return True
        except ValueError:
            if new_value in ["-", "+", ".", "e", "E"]:
                return True
            parts = new_value.split("e") if "e" in new_value else new_value.split("E")
            if len(parts) == 2 and parts[0] and parts[1].lstrip("-+") == "":
                return True
            try:
                float(new_value)
                return True
            except ValueError:
                return False

class IntegerEntry(ttk.Entry):
    def __init__(self, master=None, **kwargs):
        ttk.Entry.__init__(self, master, **kwargs)
        self.valid = (self.register(self._validate_integer), '%P')
        self.config(validate='key', validatecommand=self.valid)
    def _validate_integer(self, new_value):
        if new_value == "":
            return True
        try:
            int(new_value)
            return True
        except ValueError:
            if new_value in ["-", "+"]:
                return True
            return False

class OddIntegerEntry(ttk.Entry):
    def __init__(self, master=None, **kwargs):
        ttk.Entry.__init__(self, master, **kwargs)
        self.valid = (self.register(self._validate_odd_integer), '%P')
        self.config(validate='focusout', validatecommand=self.valid)
    def _validate_odd_integer(self, new_value):
        if new_value == "":
            return True
        try:
            value = int(new_value)
            if value % 2 != 0:
                return True
            else:
                corrected_value = value + 1
                self.delete(0, tk.END)
                self.insert(0, corrected_value)
                return True
        except ValueError:
            messagebox.showerror("Error", "Value must be a valid integer.")
            self.delete(0, tk.END)
            return False

class HoloTrackerApp:
    def __init__(self, root, controller=None):
        self.root = root
        self.root.title("HoloTracker LOCATE")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.controller = controller
        self.parameters = {}
        self.status_var = tk.StringVar()
        self.paned_window = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill="both", expand=True)
        self.left_frame = ttk.Frame(self.paned_window, width=380)
        self.paned_window.add(self.left_frame, weight=0)
        self.right_frame = ttk.Frame(self.paned_window)
        self.paned_window.add(self.right_frame, weight=1)
        self.init_left_frame()
        self.init_right_frame()

    def init_left_frame(self):
        self.left_notebook = ttk.Notebook(self.left_frame)
        self.left_notebook.pack(fill="both", expand=True)
        self.tab_path = ttk.Frame(self.left_notebook)
        self.left_notebook.add(self.tab_path, text="PATH")
        self.init_path_tab()
        self.tab_parameters = ttk.Frame(self.left_notebook)
        self.left_notebook.add(self.tab_parameters, text="PARAMETERS")
        self.init_parameters_tab()
        self.tab_actions = ttk.Frame(self.left_notebook)
        self.left_notebook.add(self.tab_actions, text="ACTIONS")
        self.init_actions_tab()

    def init_path_tab(self):
        # Holograms directory
        ttk.Label(self.tab_path, text="Holograms directory:").grid(row=0, column=0, sticky="w", pady=5)
        dir_frame = ttk.Frame(self.tab_path)
        dir_frame.grid(row=0, column=1, sticky="ew", pady=5)
        self.dir_text = tk.Text(dir_frame, height=1, width=50, wrap=tk.NONE)
        self.dir_text.grid(row=0, column=0, sticky="ew")
        dir_scroll = ttk.Scrollbar(dir_frame, orient="horizontal", command=self.dir_text.xview)
        dir_scroll.grid(row=1, column=0, sticky="ew")
        self.dir_text.configure(xscrollcommand=dir_scroll.set)
        self.browse_button = ttk.Button(self.tab_path, text="Browse", command=self.browse_directory)
        self.browse_button.grid(row=0, column=2, padx=5, pady=5)

        # Image type
        ttk.Label(self.tab_path, text="Image type:").grid(row=1, column=0, sticky="w", pady=5)
        self.image_type_var = tk.StringVar(value="TIF")
        self.image_type_combobox = ttk.Combobox(
            self.tab_path,
            textvariable=self.image_type_var,
            values=["BMP", "TIF", "JPG", "PNG"],
            state="readonly",
            width=10
        )
        self.image_type_combobox.grid(row=1, column=1, sticky="ew", pady=5)
        self.image_type_combobox.bind("<<ComboboxSelected>>", lambda e: self.on_parameter_changed("image_type", self.image_type_var.get()))

        # Mean hologram image path
        ttk.Label(self.tab_path, text="Mean hologram image path:").grid(row=2, column=0, sticky="w", pady=5)
        mean_frame = ttk.Frame(self.tab_path)
        mean_frame.grid(row=2, column=1, sticky="ew", pady=5)
        self.mean_image_text = tk.Text(mean_frame, height=1, width=50, wrap=tk.NONE)
        self.mean_image_text.grid(row=0, column=0, sticky="ew")
        mean_scroll = ttk.Scrollbar(mean_frame, orient="horizontal", command=self.mean_image_text.xview)
        mean_scroll.grid(row=1, column=0, sticky="ew")
        self.mean_image_text.configure(xscrollcommand=mean_scroll.set)
        self.browse_mean_button = ttk.Button(self.tab_path, text="Browse", command=self.browse_mean_image)
        self.browse_mean_button.grid(row=2, column=2, padx=5, pady=5)

        # Mean hologram computation button
        self.compute_mean_button = ttk.Button(self.tab_path, text="Mean hologram computation", command=self.compute_mean_hologram)
        self.compute_mean_button.grid(row=3, column=0, columnspan=3, pady=10)


    def browse_directory(self):
        directory = filedialog.askdirectory(title="Select Holograms Directory")
        if directory:
            self.dir_text.delete("1.0", tk.END)
            self.dir_text.insert(tk.END, directory)
            self.parameters["holograms_directory"] = directory
            self.on_parameter_changed("holograms_directory", directory)

    def browse_mean_image(self):
        # Autoriser uniquement les fichiers .tif et .tiff
        filetypes = [("TIFF files", "*.tif *.tiff"), ("All files", "*.*")]
        filename = filedialog.askopenfilename(
            title="Select Mean Hologram Image",
            filetypes=filetypes
        )
        if filename:
            self.mean_image_text.delete("1.0", tk.END)
            self.mean_image_text.insert(tk.END, filename)

    def compute_mean_hologram(self):
        # Show popup dialog
        result = messagebox.askyesno(
            "Mean Hologram Computation",
            "Do you want to compute mean hologram on all available holograms in directory?",
            icon="question"
        )
        
        if result:
            directory = self.dir_text.get("1.0", tk.END).strip()
        else:
            directory = filedialog.askdirectory(title="Select Directory for Mean Hologram Computation")
        
        if directory and self.controller:
            self.controller.start_mean_hologram_computation(directory, self.image_type_var.get())

    def init_parameters_tab(self):
        holo_frame = ttk.LabelFrame(self.tab_parameters, text="Holo parameters")
        holo_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.wavelength_entry = self._add_label_entry(holo_frame, "wavelength (m):", 0)
        self.optical_index_entry = self._add_label_entry(holo_frame, "optical index:", 1)
        self.holo_size_x_entry = self._add_label_entry(holo_frame, "Holo size X (pix):", 2)
        self.holo_size_y_entry = self._add_label_entry(holo_frame, "Holo size Y (pix):", 3)
        self.pixel_size_entry = self._add_label_entry(holo_frame, "pixel size (m):", 4)
        self.objective_magnification_entry = self._add_label_entry(holo_frame, "objective magnification:", 5)
        analyse_frame = ttk.LabelFrame(self.tab_parameters, text="Analyse parameters")
        analyse_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        propagation_frame = ttk.LabelFrame(analyse_frame, text="propagation")
        propagation_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.distance_entry = self._add_label_entry(propagation_frame, "distance (m):", 0)
        self.number_of_planes_entry = self._add_label_entry(propagation_frame, "number of planes:", 1)
        self.step_entry = self._add_label_entry(propagation_frame, "step (m):", 2)
        fourier_frame = ttk.LabelFrame(analyse_frame, text="Fourier BP filter (pix)")
        fourier_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        self.high_pass_entry = self._add_label_entry(fourier_frame, "High pass (0=none) (pix):", 0, IntegerEntry)
        self.low_pass_entry = self._add_label_entry(fourier_frame, "Low Pass (0=none) (pix):", 1, IntegerEntry)
        focus_frame = ttk.LabelFrame(analyse_frame, text="Focus")
        focus_frame.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")
        ttk.Label(focus_frame, text="Focus type:").grid(row=0, column=0, sticky="w")
        self.focus_type_combobox = ttk.Combobox(focus_frame, values=["SUM_OF_INTENSITY", "SUM_OF_LAPLACIAN", "SUM_OF_VARIANCE", "TENEGRAD"], width=25)
        self.focus_type_combobox.grid(row=0, column=1, sticky="w")
        self.focus_type_combobox.last_value = None
        
        def on_focus_type_change(event):
            current_value = self.focus_type_combobox.get()
            if current_value != self.focus_type_combobox.last_value:
                self.focus_type_combobox.last_value = current_value
                self.on_parameter_changed("focus_type", current_value)
        
        self.focus_type_combobox.bind("<<ComboboxSelected>>", on_focus_type_change)
        self.sum_size_entry = OddIntegerEntry(focus_frame, width=10)
        ttk.Label(focus_frame, text="Sum size:").grid(row=1, column=0, sticky="w")
        self.sum_size_entry.grid(row=1, column=1, sticky="w")
        self.sum_size_entry.bind("<FocusOut>", lambda e: self.on_parameter_changed("focus_smooth_size", self.sum_size_entry.get()))
        ttk.Label(analyse_frame, text="Remove mean Hologram:").grid(row=2, column=0, sticky="w")
        self.remove_mean_check = ttk.Checkbutton(analyse_frame, text="Off/On", command=self.on_remove_mean_changed)
        self.remove_mean_check.grid(row=2, column=1, sticky="w")
        self.remove_mean_check.last_value = None
        ttk.Label(analyse_frame, text="Batch threshold:").grid(row=3, column=1, sticky="w")
        self.batch_threshold_combobox = ttk.Combobox(analyse_frame, values=["compute on 1st hologram", "compute on each hologram"], width=25)
        self.batch_threshold_combobox.grid(row=3, column=1, sticky="w")
        self.batch_threshold_combobox.last_value = None
        
        def on_batch_threshold_change(event):
            current_value = self.batch_threshold_combobox.get()
            if current_value != self.batch_threshold_combobox.last_value:
                self.batch_threshold_combobox.last_value = current_value
                self.on_parameter_changed("recalc_threshold", current_value == "compute on each hologram")
        
        self.batch_threshold_combobox.bind("<<ComboboxSelected>>", on_batch_threshold_change)
        ccl_frame = ttk.LabelFrame(analyse_frame, text="CCL parameters")
        ccl_frame.grid(row=4, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")
        self.threshold_entry = self._add_label_entry(ccl_frame, "Threshold (N x Standard deviation):", 0, IntegerEntry)
        ttk.Label(ccl_frame, text="Connectivity:").grid(row=1, column=0, sticky="w")
        self.connectivity_combobox = ttk.Combobox(ccl_frame, values=["6", "18", "26"], width=10)
        self.connectivity_combobox.grid(row=1, column=1, sticky="w")
        self.connectivity_combobox.last_value = None
        
        def on_connectivity_change(event):
            current_value = self.connectivity_combobox.get()
            if current_value != self.connectivity_combobox.last_value:
                self.connectivity_combobox.last_value = current_value
                self.on_parameter_changed("n_connectivity", current_value)
        
        self.connectivity_combobox.bind("<<ComboboxSelected>>", on_connectivity_change)
        self.min_voxel_entry = self._add_label_entry(ccl_frame, "min voxel:", 2, IntegerEntry)
        self.max_voxel_entry = self._add_label_entry(ccl_frame, "max voxel:", 3, IntegerEntry)

    def _add_label_entry(self, parent, text, row, entry_class=ScientificFloatEntry):
        ttk.Label(parent, text=text).grid(row=row, column=0, sticky="w")
        entry = entry_class(parent, width=10)
        entry.grid(row=row, column=1, sticky="w")
        
        # Map UI label names to parameter names used in core
        param_name_map = {
            "wavelength": "medium_wavelength",
            "optical": "optical_index", 
            "Holo": "holo_size_x" if "X" in text else "holo_size_y" if "Y" in text else "holo_size",
            "pixel": "cam_pix_size",
            "objective": "cam_magnification",
            "distance": "distance_z0",
            "number": "nb_plan_reconstruit",
            "step": "dz",
            "High": "high_pass",
            "Low": "low_pass",
            "Threshold": "nb_StdVar_threshold",
            "min": "min_voxel",
            "max": "max_voxel"
        }
        
        # Find matching parameter name
        param_name = text.split()[0]
        for key, value in param_name_map.items():
            if key in text:
                param_name = value
                break
        
        # Store initial value for change detection
        entry.last_value = None
        
        def on_value_change(event, name=param_name):
            current_value = entry.get()
            if current_value != entry.last_value:
                entry.last_value = current_value
                self.on_parameter_changed(name, current_value)
                
        entry.bind("<FocusOut>", on_value_change)
        entry.bind("<KeyPress-Return>", on_value_change)
        return entry

    def on_parameter_changed(self, name, value):
        if self.controller:
            self.controller.on_parameter_changed(name, value)
        if name in ["holograms_directory", "image_type"]:
            self.update_hologram_list()

    def on_remove_mean_changed(self):
        if self.controller:
            current_value = self.remove_mean_check.instate(['selected'])
            if current_value != self.remove_mean_check.last_value:
                self.remove_mean_check.last_value = current_value
                self.controller.on_parameter_changed("remove_mean", current_value)

    def init_actions_tab(self):
        display_options = [
            "RAW_HOLOGRAM",
            "CLEANED_HOLOGRAM",
            "FILTERED_HOLOGRAM",
            "VOLUME_PLANE_NUMBER",
            "XY_SUM_PROJECTION",
            "XZ_SUM_PROJECTION",
            "YZ_SUM_PROJECTION",
            "XY_MAX_PROJECTION",
            "XZ_MAX_PROJECTION",
            "YZ_MAX_PROJECTION"
        ]
        ttk.Label(self.tab_actions, text="Display:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.display_combobox = ttk.Combobox(self.tab_actions, values=display_options, width=20)
        self.display_combobox.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        self.display_combobox.last_value = None
        self.display_combobox.bind("<<ComboboxSelected>>", self.on_display_changed)
        ttk.Label(self.tab_actions, text="Plane number:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.plane_number_spinbox = ttk.Spinbox(self.tab_actions, from_=0, to=100, width=5, command=self.on_plane_number_changed)
        self.plane_number_spinbox.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        self.plane_number_spinbox.last_value = None
        self.plane_number_spinbox.bind("<Return>", lambda e: self.on_plane_number_changed())
        self.plane_number_spinbox.bind("<FocusOut>", lambda e: self.on_plane_number_changed())
        # ==================== TEST MODE SECTION ====================
        test_frame = ttk.LabelFrame(self.tab_actions, text="Test Mode", padding=5)
        test_frame.grid(row=2, column=0, columnspan=2, pady=10, padx=5, sticky="ew")
        
        ttk.Label(test_frame, text="hologram to test:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.hologram_combobox = ttk.Combobox(test_frame, width=20)
        self.hologram_combobox.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        self.hologram_combobox.bind("<<ComboboxSelected>>", self.on_hologram_selected)
        self.enter_test_button = ttk.Button(test_frame, text="ENTER IN TEST MODE", command=self.on_enter_test_mode)
        self.enter_test_button.grid(row=1, column=0, pady=10, padx=5)
        self.exit_test_button = ttk.Button(test_frame, text="EXIT TEST MODE", command=self.on_exit_test_mode)
        self.exit_test_button.grid(row=1, column=1, pady=10, padx=5)
        # ==================== BATCH PROCESSING SECTION ====================
        batch_frame = ttk.LabelFrame(self.tab_actions, text="Batch Processing", padding=5)
        batch_frame.grid(row=3, column=0, columnspan=2, pady=10, padx=5, sticky="ew")
        
        # Batch control buttons
        button_frame = ttk.Frame(batch_frame)
        button_frame.grid(row=0, column=0, columnspan=2, pady=5, sticky="ew")
        
        self.start_batch_button = ttk.Button(button_frame, text="START BATCH", command=self.on_start_batch)
        self.start_batch_button.grid(row=0, column=0, padx=5)
        self.stop_batch_button = ttk.Button(button_frame, text="STOP BATCH", command=self.on_stop_batch)
        self.stop_batch_button.grid(row=0, column=1, padx=5)
        
        # Display results checkbox
        self.display_batch_results_var = tk.BooleanVar(value=False)
        self.display_batch_results_check = ttk.Checkbutton(
            button_frame, 
            text="DISPLAY RESULTS", 
            variable=self.display_batch_results_var
        )
        self.display_batch_results_check.grid(row=0, column=2, padx=10)
        
        # CSV output info
        self.batch_csv_var = tk.StringVar(value="")
        self.batch_csv_label = ttk.Label(batch_frame, textvariable=self.batch_csv_var, foreground="blue", font=("Arial", 8))
        self.batch_csv_label.grid(row=1, column=0, columnspan=3, sticky="w", pady=2)
        
        # Configure grid weights for proper resizing
        batch_frame.columnconfigure(1, weight=1)
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)

        # Zone d'affichage des temps d'exécution
        self.timing_frame = ttk.Frame(self.tab_actions)
        self.timing_frame.grid(row=4, column=0, columnspan=2, pady=10, padx=5, sticky="ew")
        
        ttk.Label(self.timing_frame, text="Processing time:", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky="w")
        self.timing_label = ttk.Label(self.timing_frame, text="No processing performed", font=("Arial", 9), foreground="gray")
        self.timing_label.grid(row=1, column=0, sticky="w")

        style = ttk.Style()
        style.configure("Red.TButton", foreground="white", background="#d9534f")
        self.exit_button = ttk.Button(self.tab_actions, text="EXIT", bootstyle="danger", command=self.on_closing)
        self.exit_button.grid(row=5, column=0, columnspan=2, pady=10)

    def update_buttons_state(self, state):
        # WAIT: only allow ENTER TEST MODE and START BATCH
        # TEST_MODE: only allow EXIT TEST MODE
        # BATCH_MODE: only allow STOP BATCH
        if state == "WAIT":
            self.enter_test_button.config(state="normal")
            self.exit_test_button.config(state="disabled")
            self.start_batch_button.config(state="normal")
            self.stop_batch_button.config(state="disabled")
        elif state == "TEST_MODE":
            self.enter_test_button.config(state="disabled")
            self.exit_test_button.config(state="normal")
            self.start_batch_button.config(state="disabled")
            self.stop_batch_button.config(state="disabled")
        elif state == "BATCH_MODE":
            self.enter_test_button.config(state="disabled")
            self.exit_test_button.config(state="disabled")
            self.start_batch_button.config(state="disabled")
            self.stop_batch_button.config(state="normal")
        else:
            # Default: enable all basic controls
            self.enter_test_button.config(state="normal")
            self.exit_test_button.config(state="normal")
            self.start_batch_button.config(state="normal")
            self.stop_batch_button.config(state="normal")

    def on_display_changed(self, event=None):
        """Called when display type changes"""
        if self.controller:
            display_type = self.display_combobox.get()
            plane_number = int(self.plane_number_spinbox.get()) if self.plane_number_spinbox.get() else 0
            
            # Only trigger if display type actually changed
            if display_type != self.display_combobox.last_value:
                self.display_combobox.last_value = display_type
                self.controller.on_display_changed(display_type, plane_number)

    def on_plane_number_changed(self):
        """Called when plane number changes"""
        if self.controller:
            display_type = self.display_combobox.get()
            plane_number = int(self.plane_number_spinbox.get()) if self.plane_number_spinbox.get() else 0
            
            # Only trigger if plane number actually changed
            if plane_number != self.plane_number_spinbox.last_value:
                self.plane_number_spinbox.last_value = plane_number
                # Only update if we're in VOLUME_PLANE_NUMBER mode
                if display_type == "VOLUME_PLANE_NUMBER":
                    self.controller.on_display_changed(display_type, plane_number)

    def set_default_display_type(self, remove_mean=False):
        """Set display type based on remove_mean parameter"""
        if remove_mean:
            self.display_combobox.set("CLEANED_HOLOGRAM")
        else:
            self.display_combobox.set("RAW_HOLOGRAM")

    def on_enter_test_mode(self):
        if self.controller:
            self.controller.on_enter_test_mode()

    def on_exit_test_mode(self):
        if self.controller:
            self.controller.on_exit_test_mode()

    # ==================== BATCH PROCESSING METHODS ====================
    
    def on_start_batch(self):
        """Start batch processing mode"""
        if self.controller:
            # Use the hologram directory from PATH tab
            hologram_directory = self.dir_text.get("1.0", "end-1c").strip()
            if not hologram_directory:
                messagebox.showwarning("Warning", "Please select a hologram directory first in the PATH tab.")
                return
            if not os.path.exists(hologram_directory):
                messagebox.showerror("Error", "Selected hologram directory does not exist.")
                return
            
            # Start batch mode through controller
            self.controller.on_enter_batch_mode(hologram_directory)
    
    def on_stop_batch(self):
        """Stop batch processing mode"""
        if self.controller:
            self.controller.on_exit_batch_mode()
    
    def on_batch_mode_entered(self, csv_filename):
        """Called when batch mode is successfully entered"""
        self.batch_csv_var.set(f"CSV Output: {csv_filename}")
        
        # Gray out parameter controls during batch
        self.set_parameters_enabled(False)
        
    def on_batch_mode_exited(self):
        """Called when batch mode is exited""" 
        self.batch_csv_var.set("")
        
        # Re-enable parameter controls
        self.set_parameters_enabled(True)
    
    def update_batch_progress(self, hologram_number):
        """Update batch processing progress - now shows in status bar"""
        pass  # Progress is now handled by the status bar
    
    def set_parameters_enabled(self, enabled):
        """Enable/disable parameter controls during batch processing"""
        state = "normal" if enabled else "disabled"
        combobox_state = "readonly" if enabled else "disabled"
        
        # PATH TAB - Directory controls
        self.browse_button.config(state=state)
        self.browse_mean_button.config(state=state)
        self.dir_text.config(state=state)
        self.image_type_combobox.config(state=combobox_state)
        self.mean_image_text.config(state=state)
        self.compute_mean_button.config(state=state)
        
        # PARAMETERS TAB - All hologram and analysis parameters
        # Holo parameters
        self.wavelength_entry.config(state=state)
        self.optical_index_entry.config(state=state)
        self.holo_size_x_entry.config(state=state)
        self.holo_size_y_entry.config(state=state)
        self.pixel_size_entry.config(state=state)
        self.objective_magnification_entry.config(state=state)
        
        # Propagation parameters
        self.distance_entry.config(state=state)
        self.number_of_planes_entry.config(state=state)
        self.step_entry.config(state=state)
        
        # Fourier filter parameters
        self.high_pass_entry.config(state=state)
        self.low_pass_entry.config(state=state)
        
        # Focus parameters
        self.focus_type_combobox.config(state=combobox_state)
        self.sum_size_entry.config(state=state)
        
        # Analysis parameters
        self.remove_mean_check.config(state=state)
        self.batch_threshold_combobox.config(state=combobox_state)
        
        # CCL parameters
        self.threshold_entry.config(state=state)
        self.connectivity_combobox.config(state=combobox_state)
        self.min_voxel_entry.config(state=state)
        self.max_voxel_entry.config(state=state)
        
        # ACTIONS TAB - Test mode controls (keep DISPLAY active for dynamic changes)
        self.hologram_combobox.config(state=combobox_state)
        self.enter_test_button.config(state=state)
        # Note: DISPLAY and plane controls remain enabled for dynamic changes during batch
    
    # Legacy methods for backward compatibility
    def on_batch_process(self):
        """Legacy method - redirect to new batch system"""
        self.on_start_batch()

    def on_cancel_batch(self):
        """Legacy method - redirect to new batch system"""
        self.on_stop_batch()

    def init_right_frame(self):
        self.right_notebook = ttk.Notebook(self.right_frame)
        self.right_notebook.pack(fill="both", expand=True)
        self.tab_hologram = ttk.Frame(self.right_notebook)
        self.right_notebook.add(self.tab_hologram, text="Hologram")
        self.init_hologram_tab()
        self.tab_localisations = ttk.Frame(self.right_notebook)
        self.right_notebook.add(self.tab_localisations, text="Localisations")
        self.init_localisations_tab()
        self.tab_infos = ttk.Frame(self.right_notebook)
        self.right_notebook.add(self.tab_infos, text="Infos")
        self.init_infos_tab()
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_label = ttk.Label(self.right_frame, textvariable=self.status_var, relief="sunken", anchor="w")
        self.status_label.pack(fill="x", padx=2, pady=2)

    def init_hologram_tab(self):
        image_frame = ttk.Frame(self.tab_hologram)
        image_frame.pack(fill="both", expand=True)

        image_frame.grid_rowconfigure(0, weight=1)
        image_frame.grid_columnconfigure(0, weight=1)

        self.image_canvas = tk.Canvas(image_frame, bg="#cccccc")
        self.image_canvas.grid(row=0, column=0, columnspan=4, sticky="nsew")

        x_scroll = ttk.Scrollbar(image_frame, orient="horizontal", command=self.image_canvas.xview)
        y_scroll = ttk.Scrollbar(image_frame, orient="vertical", command=self.image_canvas.yview)
        self.image_canvas.configure(xscrollcommand=x_scroll.set, yscrollcommand=y_scroll.set)
        x_scroll.grid(row=1, column=0, columnspan=4, sticky="ew")
        y_scroll.grid(row=0, column=4, sticky="ns")

        btn_width = 12
        ttk.Button(image_frame, text="Zoom +", width=btn_width, command=self.zoom_in).grid(row=2, column=0, sticky="ew", pady=5)
        ttk.Button(image_frame, text="Zoom -", width=btn_width, command=self.zoom_out).grid(row=2, column=1, sticky="ew", pady=5)
        ttk.Button(image_frame, text="Stretch", width=btn_width, command=self.stretch_zoom).grid(row=2, column=2, sticky="ew", pady=5)
        ttk.Button(image_frame, text="Reset zoom", width=btn_width, command=self.reset_zoom).grid(row=2, column=3, sticky="ew", pady=5)

        self.zoom_mode = "real"  # "real" or "stretch"
        self.zoom_level = 1.0
        self.pil_image = None
        self.tk_img = None

        self.image_canvas.bind("<Configure>", self._on_canvas_resize)

    def display_hologram_image(self, pil_image):
        self.pil_image = pil_image
        self.zoom_mode = "real"
        self.zoom_level = 1.0
        self._show_image()

    def _show_image(self):
        if self.pil_image:
            img_w, img_h = self.pil_image.size
            if self.zoom_mode == "stretch":
                canvas_w = self.image_canvas.winfo_width()
                canvas_h = self.image_canvas.winfo_height()
                scale = min(canvas_w / img_w, canvas_h / img_h)
                new_w, new_h = int(img_w * scale), int(img_h * scale)
            else:  # "real" mode
                new_w, new_h = int(img_w * self.zoom_level), int(img_h * self.zoom_level)
            img = self.pil_image.resize((new_w, new_h))
            self.tk_img = ImageTk.PhotoImage(img)
            self.image_canvas.delete("all")
            self.image_canvas.create_image(0, 0, anchor="nw", image=self.tk_img)
            self.image_canvas.config(scrollregion=(0, 0, new_w, new_h))

    def zoom_in(self):
        # If just switched from stretch, start zoom from current displayed size
        if self.zoom_mode == "stretch":
            self.zoom_mode = "real"
            # Calculate zoom_level based on current canvas/image size
            img_w, img_h = self.pil_image.size
            canvas_w = self.image_canvas.winfo_width()
            canvas_h = self.image_canvas.winfo_height()
            scale = min(canvas_w / img_w, canvas_h / img_h)
            self.zoom_level = scale
        self.zoom_level *= 1.2
        self._show_image()

    def zoom_out(self):
        if self.zoom_mode == "stretch":
            self.zoom_mode = "real"
            img_w, img_h = self.pil_image.size
            canvas_w = self.image_canvas.winfo_width()
            canvas_h = self.image_canvas.winfo_height()
            scale = min(canvas_w / img_w, canvas_h / img_h)
            self.zoom_level = scale
        self.zoom_level /= 1.2
        self._show_image()

    def stretch_zoom(self):
        self.zoom_mode = "stretch"
        self._show_image()

    def reset_zoom(self):
        self.zoom_mode = "real"
        self.zoom_level = 1.0
        self._show_image()

    def _on_canvas_resize(self, event):
        if self.zoom_mode == "stretch":
            self._show_image()

    def init_localisations_tab(self):
        self.fig = Figure(figsize=(6,6), dpi=100)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlabel('X Position (µm)')
        self.ax.set_ylabel('Y Position (µm)') 
        self.ax.set_zlabel('Z Position (µm)')
        self.ax.set_title('Particle Localization')
        self.colorbar = None  # Initialize colorbar
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.tab_localisations)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, self.tab_localisations)
        toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def display_results(self, results):
        """Update the 3D plot in the Localisation tab"""
        print(f"🖼️  UI: display_results called with keys: {list(results.keys()) if results else 'None'}")
        self.ax.clear()
        
        if "localizations" in results and results["localizations"]:
            x, y, z = zip(*results["localizations"])
            
            # Use particle sizes for coloring if available
            if "particle_sizes" in results and results["particle_sizes"]:
                colors = results["particle_sizes"]
                scatter = self.ax.scatter(x, y, z, c=colors, cmap='viridis', marker='o', s=50)
                
                # Add colorbar if not already present
                try:
                    if not hasattr(self, 'colorbar') or self.colorbar is None:
                        self.colorbar = self.fig.colorbar(scatter, ax=self.ax, label='Particle Size (pixels)', shrink=0.8)
                    else:
                        # Remove old colorbar and add new one
                        self.colorbar.remove()
                        self.colorbar = None  # Reset to None first
                        self.colorbar = self.fig.colorbar(scatter, ax=self.ax, label='Particle Size (pixels)', shrink=0.8)
                except Exception as e:
                    print(f"⚠️  UI: Error managing colorbar: {e}")
                    # Continue without colorbar if there's an error
            else:
                self.ax.scatter(x, y, z, c='red', marker='o', s=50)
            
            count = results.get('count', len(x))
            title = f'Particle Positions ({count} objects detected)'
        else:
            title = 'No particles detected'
            # Remove colorbar if no data
            try:
                if hasattr(self, 'colorbar') and self.colorbar is not None:
                    self.colorbar.remove()
                    self.colorbar = None
            except Exception as e:
                print(f"⚠️  UI: Error removing colorbar: {e}")
                self.colorbar = None
            
        self.ax.set_xlabel('X Position (µm)')
        self.ax.set_ylabel('Y Position (µm)') 
        self.ax.set_zlabel('Z Position (µm)')
        self.ax.set_title(title)
        
        # Mettre à jour l'affichage des temps d'exécution (sauf en mode BATCH)
        if not (hasattr(self, 'controller') and self.controller and self.controller.state == "BATCH_MODE"):
            print("📞 UI: About to call update_timing_display (not in BATCH mode)")
            self.update_timing_display(results)
            print("✅ UI: update_timing_display completed")
        else:
            print("🔄 UI: Skipping update_timing_display (BATCH mode - controller handles it)")
        
        # Update Info tab table
        self.update_results_table(results)
        
        # Force canvas refresh
        self.canvas.draw()
        self.canvas.flush_events()

    def update_timing_display(self, results):
        """Met à jour l'affichage des temps d'exécution dans l'onglet Actions"""
        print(f"🕐 UI: update_timing_display called with keys: {list(results.keys()) if results else 'None'}")
        
        if hasattr(self, 'timing_label'):
            print("✅ UI: timing_label exists")
            if 'processing_times' in results and results['processing_times']:
                times = results['processing_times']
                print(f"⏱️  UI: Processing times found: {times}")
                print(f"🔍 UI: Keys in times: {list(times.keys())}")
                print(f"🔍 UI: Does 'iteration_time' exist? {'iteration_time' in times}")
                timing_text = ""
                
                def format_time(t):
                    """Formate le temps en ms si < 1s, sinon en s"""
                    if t < 1.0:
                        return f"{t*1000:.1f}ms"
                    else:
                        return f"{t:.3f}s"
                
                # Affichage détaillé des temps selon les données disponibles
                if 'preprocessing' in times:
                    timing_text += f"Preprocessing: {format_time(times['preprocessing'])}\n"
                if 'filtering' in times:
                    timing_text += f"Filtering: {format_time(times['filtering'])}\n"
                if 'propagation' in times:
                    timing_text += f"Propagation: {format_time(times['propagation'])}\n"
                if 'focus' in times:
                    timing_text += f"Focus détection: {format_time(times['focus'])}\n"
                if 'ccl' in times:
                    timing_text += f"CCL3D: {format_time(times['ccl'])}\n"
                if 'cca' in times:
                    timing_text += f"CCA: {format_time(times['cca'])}\n"
                if 'total_processing' in times:
                    timing_text += f"Total: {format_time(times['total_processing'])}"
                
                # Ajouter temps d'itération si présent (mode BATCH)
                if 'iteration_time' in times:
                    iteration_ms = times['iteration_time'] * 1000
                    timing_text += f"\nIteration time: {iteration_ms:.1f}ms"
                    print(f"✅ UI: Added iteration time to display: {iteration_ms:.1f}ms")
                
                print(f"📝 UI: Generated timing text: {timing_text[:50]}...")
                
                # Mise à jour avec couleur normale
                print("🖼️  UI: Updating timing_label...")
                self.timing_label.config(text=timing_text, foreground="white", background="red")
                print("✅ UI: timing_label updated successfully")
            else:
                print("❌ UI: No processing_times in results or processing_times is empty")
                # Aucun temps disponible
                self.timing_label.config(text="Aucun traitement effectué", foreground="gray")
        else:
            print("❌ UI: timing_label does not exist!")

    def update_3d_plot(self, core_results):
        """Convenience method to update 3D plot from core results"""
        if hasattr(core_results, 'get_3d_results_data'):
            results_data = core_results.get_3d_results_data()
            if results_data:
                self.display_results(results_data)
            else:
                self.display_results({'localizations': [], 'count': 0})

    def init_infos_tab(self):
        """Initialize the Info tab with object detection results table"""
        # Create main frame for info tab content
        info_frame = ttk.Frame(self.tab_infos)
        info_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Add title label
        title_label = ttk.Label(info_frame, text="Object Detection Results", 
                               font=('TkDefaultFont', 12, 'bold'))
        title_label.pack(pady=(0, 10))
        
        # Create frame for the table
        table_frame = ttk.Frame(info_frame)
        table_frame.pack(fill="both", expand=True)
        
        # Create Treeview for the results table
        columns = ("Object", "Position X", "Position Y", "Position Z", "nb_voxel")
        self.results_tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=15)
        
        # Configure column headings and widths
        self.results_tree.heading("Object", text="Object #")
        self.results_tree.heading("Position X", text="Position X (µm)")
        self.results_tree.heading("Position Y", text="Position Y (µm)")
        self.results_tree.heading("Position Z", text="Position Z (µm)")
        self.results_tree.heading("nb_voxel", text="nb_voxel")
        
        self.results_tree.column("Object", width=80, anchor="center")
        self.results_tree.column("Position X", width=120, anchor="center")
        self.results_tree.column("Position Y", width=120, anchor="center")
        self.results_tree.column("Position Z", width=120, anchor="center")
        self.results_tree.column("nb_voxel", width=100, anchor="center")
        
        # Add scrollbars
        v_scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.results_tree.yview)
        h_scrollbar = ttk.Scrollbar(table_frame, orient="horizontal", command=self.results_tree.xview)
        self.results_tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Grid layout for table and scrollbars
        self.results_tree.grid(row=0, column=0, sticky="nsew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        h_scrollbar.grid(row=1, column=0, sticky="ew")
        
        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)
        
        # Bind double-click event to show object details
        self.results_tree.bind("<Double-1>", self.on_object_double_click)
        
        # Add summary label
        self.summary_label = ttk.Label(info_frame, text="No objects detected", 
                                      font=('TkDefaultFont', 10))
        self.summary_label.pack(pady=(10, 0))

    def update_results_table(self, results):
        """Update the object detection results table in Info tab"""
        # Clear existing data
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        if "localizations" in results and results["localizations"]:
            localizations = results["localizations"]
            particle_sizes = results.get("particle_sizes", [])
            
            # Populate table with detection results
            for i, (x, y, z) in enumerate(localizations):
                object_num = i + 1
                nb_voxel = particle_sizes[i] if i < len(particle_sizes) else "N/A"
                
                # Format coordinates to 3 decimal places
                x_formatted = f"{x:.3f}"
                y_formatted = f"{y:.3f}"
                z_formatted = f"{z:.3f}"
                
                self.results_tree.insert("", "end", values=(
                    object_num, x_formatted, y_formatted, z_formatted, nb_voxel
                ))
            
            # Update summary
            count = results.get('count', len(localizations))
            self.summary_label.config(text=f"Total objects detected: {count}")
        else:
            # No objects detected
            self.summary_label.config(text="No objects detected")

    def load_parameters(self):
        if os.path.exists("last_param.json"):
            with open("last_param.json", "r") as f:
                self.parameters = json.load(f)
                self.update_ui_from_parameters()
        else:
            self.parameters = {}
        self.update_hologram_list()
    
    def save_parameters(self):
        self.update_parameters_from_ui()
        with open("last_param.json", "w") as f:
            json.dump(self.parameters, f, indent=4)

    def update_parameters_from_ui(self):
        self.parameters = {
            "mean_hologram_image_path": self.mean_image_text.get("1.0", tk.END).strip(),
            "wavelength": self.wavelength_entry.get(),
            "optical_index": self.optical_index_entry.get(),
            "holo_size_x": self.holo_size_x_entry.get(),
            "holo_size_y": self.holo_size_y_entry.get(),
            "pixel_size": self.pixel_size_entry.get(),
            "objective_magnification": self.objective_magnification_entry.get(),
            "distance": self.distance_entry.get(),
            "number_of_planes": self.number_of_planes_entry.get(),
            "step": self.step_entry.get(),
            "high_pass": self.high_pass_entry.get(),
            "low_pass": self.low_pass_entry.get(),
            "focus_type": self.focus_type_combobox.get(),
            "sum_size": self.sum_size_entry.get(),
            "remove_mean": self.remove_mean_check.instate(['selected']),
            "batch_threshold": self.batch_threshold_combobox.get(),
            "nb_StdVar_threshold": self.threshold_entry.get(),
            "connectivity": self.connectivity_combobox.get(),
            "min_voxel": self.min_voxel_entry.get(),
            "max_voxel": self.max_voxel_entry.get(),
            "holograms_directory": self.dir_text.get("1.0", tk.END).strip(),
            "image_type": self.image_type_var.get()
        }

    def update_ui_from_parameters(self):
        if not self.parameters: return
        for attr, value in self.parameters.items():
            # Handle special mapping for core parameter names
            widget_name = attr
            if attr == "nb_StdVar_threshold":
                widget_name = "threshold"
            
            widget = getattr(self, f"{widget_name}_entry", None)
            if widget:
                widget.delete(0, tk.END)
                widget.insert(0, value)
                # Update last_value for change detection
                if hasattr(widget, 'last_value'):
                    widget.last_value = widget.get()
            combobox = getattr(self, f"{widget_name}_combobox", None)
            if combobox:
                combobox.set(value)
                # Update last_value for change detection
                if hasattr(combobox, 'last_value'):
                    combobox.last_value = combobox.get()
        if self.parameters.get("remove_mean", False):
            self.remove_mean_check.state(['selected'])
        else:
            self.remove_mean_check.state(['!selected'])
        # Update last_value for remove_mean checkbox
        if hasattr(self.remove_mean_check, 'last_value'):
            self.remove_mean_check.last_value = self.remove_mean_check.instate(['selected'])
            
        if "holograms_directory" in self.parameters:
            self.dir_text.delete(1.0, tk.END)
            self.dir_text.insert(tk.END, self.parameters["holograms_directory"])
        if "image_type" in self.parameters:
            self.image_type_var.set(self.parameters["image_type"])
        if "mean_hologram_image_path" in self.parameters:
            self.mean_image_text.delete(1.0, tk.END)
            self.mean_image_text.insert(tk.END, self.parameters["mean_hologram_image_path"])
            
        # Update last_value for display and plane widgets in actions tab
        if hasattr(self, 'display_combobox') and hasattr(self.display_combobox, 'last_value'):
            self.display_combobox.last_value = self.display_combobox.get()
        if hasattr(self, 'plane_number_spinbox') and hasattr(self.plane_number_spinbox, 'last_value'):
            try:
                self.plane_number_spinbox.last_value = int(self.plane_number_spinbox.get()) if self.plane_number_spinbox.get() else 0
            except:
                self.plane_number_spinbox.last_value = 0

    def on_closing(self):
        # Only allow closing if Exit button is enabled
        if str(self.exit_button['state']) == "disabled":
            return  # Ignore close request
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.save_parameters()
            self.root.destroy()

    def update_hologram_list(self):
        """Update the 'hologram to test' dropdown with image files from the selected directory and type."""
        directory = self.dir_text.get("1.0", tk.END).strip()
        image_type = self.image_type_var.get().lower()
        ext_map = {
            "bmp": ".bmp",
            "tif": ".tif",
            "jpg": ".jpg",
            "png": ".png"
        }
        ext = ext_map.get(image_type, ".tif")
        if os.path.isdir(directory):
            image_files = [f for f in os.listdir(directory) if f.lower().endswith(ext)]
        else:
            image_files = []
        self.hologram_combobox['values'] = image_files
        if image_files:
            self.hologram_combobox.set(image_files[0])
        else:
            self.hologram_combobox.set("")

    def on_hologram_selected(self, event=None):
        print(f"🎯 UI: on_hologram_selected called")
        directory = self.dir_text.get("1.0", tk.END).strip()
        filename = self.hologram_combobox.get()
        if self.controller:
            print(f"📞 UI: Calling controller.on_hologram_selected({filename})")
            self.controller.on_hologram_selected(directory, filename)
            # SUPPRIMÉ: Auto-update display - c'est déjà géré dans on_hologram_selected du controller

    def set_buttons_state_for_mean_computation(self, computing):
        """Enable/disable buttons during mean computation"""
        state = "disabled" if computing else "normal"
        self.browse_button.config(state=state)
        self.browse_mean_button.config(state=state)
        self.compute_mean_button.config(state=state)
        self.image_type_combobox.config(state="disabled" if computing else "readonly")
        # Also disable action buttons
        if hasattr(self, 'enter_test_button'):
            self.enter_test_button.config(state=state)
            self.exit_test_button.config(state=state)
            self.batch_process_button.config(state=state)

    def on_object_double_click(self, event):
        """Handle double-click on an object in the results table"""
        selection = self.results_tree.selection()
        if not selection:
            return
            
        # Get selected item data
        item = self.results_tree.selection()[0]
        values = self.results_tree.item(item, "values")
        if not values:
            return
            
        object_num = int(values[0])
        pos_x = float(values[1])
        pos_y = float(values[2])
        pos_z = float(values[3])
        nb_voxel = values[4]
        
        print(f"🔍 Object #{object_num} clicked: ({pos_x:.3f}, {pos_y:.3f}, {pos_z:.3f}) - {nb_voxel} voxels")
        
        # Show parameter dialog
        self.show_object_viewer_dialog(object_num, pos_x, pos_y, pos_z)

    def show_object_viewer_dialog(self, object_num, pos_x, pos_y, pos_z):
        """Show dialog to get viewer parameters and display object slices"""
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Object #{object_num} Viewer Parameters")
        dialog.geometry("300x200")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Center the dialog
        dialog.geometry("+%d+%d" % (self.root.winfo_rootx() + 50, self.root.winfo_rooty() + 50))
        
        # Create input fields
        ttk.Label(dialog, text=f"Object #{object_num} at ({pos_x:.1f}, {pos_y:.1f}, {pos_z:.1f})").pack(pady=10)
        
        # Vox_size_XY field
        xy_frame = ttk.Frame(dialog)
        xy_frame.pack(pady=5)
        ttk.Label(xy_frame, text="Vox_size_XY:").pack(side="left")
        xy_var = tk.StringVar(value="20")
        xy_entry = ttk.Entry(xy_frame, textvariable=xy_var, width=10)
        xy_entry.pack(side="left", padx=(10, 0))
        
        # Vox_size_Z field
        z_frame = ttk.Frame(dialog)
        z_frame.pack(pady=5)
        ttk.Label(z_frame, text="Vox_size_Z:").pack(side="left")
        z_var = tk.StringVar(value="20")
        z_entry = ttk.Entry(z_frame, textvariable=z_var, width=10)
        z_entry.pack(side="left", padx=(10, 0))
        
        # Button frame
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=20)
        
        def on_ok():
            try:
                vox_xy = int(xy_var.get())
                vox_z = int(z_var.get())
                if vox_xy <= 0 or vox_z <= 0:
                    raise ValueError("Sizes must be positive")
                dialog.destroy()
                self.show_object_slices(object_num, pos_x, pos_y, pos_z, vox_xy, vox_z)
            except ValueError as e:
                tk.messagebox.showerror("Invalid Input", "Please enter valid positive integers")
        
        def on_cancel():
            dialog.destroy()
        
        ttk.Button(button_frame, text="OK", command=on_ok).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(side="left", padx=5)
        
        # Focus on first entry and select all
        xy_entry.focus()
        xy_entry.select_range(0, "end")

    def show_object_slices(self, object_num, pos_x, pos_y, pos_z, vox_xy, vox_z):
        """Extract and display 3 slice views of the object from the reconstructed volume"""
        if not self.controller:
            tk.messagebox.showerror("Error", "No controller available")
            return
            
        # Request slice extraction from controller
        self.controller.extract_object_slices(object_num, pos_x, pos_y, pos_z, vox_xy, vox_z)

    def display_object_slices(self, object_num, slices_data):
        """Display the 3 slice views in a new window"""
        viewer_window = tk.Toplevel(self.root)
        viewer_window.title(f"Object #{object_num} - Slice Views")
        viewer_window.geometry("900x350")
        
        # Create frame for the 3 images
        main_frame = ttk.Frame(viewer_window)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        slice_names = ["XY Plane", "XZ Plane", "YZ Plane"]
        slice_keys = ["xy_slice", "xz_slice", "yz_slice"]
        
        for i, (name, key) in enumerate(zip(slice_names, slice_keys)):
            if key not in slices_data:
                continue
                
            # Create frame for this slice
            slice_frame = ttk.LabelFrame(main_frame, text=name)
            slice_frame.grid(row=0, column=i, padx=5, pady=5, sticky="nsew")
            
            # Get slice data
            slice_array = slices_data[key]
            
            # Convert to PIL Image and then to PhotoImage
            # Normalize to 0-255 range
            slice_normalized = ((slice_array - slice_array.min()) / 
                              (slice_array.max() - slice_array.min() + 1e-8) * 255).astype(np.uint8)
            
            # Create PIL Image
            pil_image = Image.fromarray(slice_normalized, mode='L')
            
            # Scale up for better visibility (2x)
            new_size = (pil_image.width * 2, pil_image.height * 2)
            pil_image = pil_image.resize(new_size, Image.NEAREST)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(pil_image)
            
            # Create label to display image
            image_label = ttk.Label(slice_frame, image=photo)
            image_label.image = photo  # Keep a reference
            image_label.pack(padx=5, pady=5)
            
            # Add size info
            info_label = ttk.Label(slice_frame, text=f"Size: {slice_array.shape}")
            info_label.pack()
        
        # Configure grid weights
        for i in range(3):
            main_frame.grid_columnconfigure(i, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)
