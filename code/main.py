import threading
import queue
import os
import numpy as np
import tkinter as tk  # Add this import
from PIL import Image
import ttkbootstrap as tb
from ui import HoloTrackerApp
from core import HoloTrackerCore
from controller_threaded import HoloTrackerController

root = tb.Window(themename="superhero")  # Utilise le thème sombre "darkly"

core = HoloTrackerCore()
app_ui = HoloTrackerApp(root)
controller = HoloTrackerController(app_ui, core)
app_ui.controller = controller  # Inject the controller into the UI

# Load parameters and synchronize them properly
app_ui.load_parameters()
controller.sync_parameters_to_core()

try:
    root.mainloop()
finally:
    # Cleanup lors de la fermeture
    controller.cleanup()
