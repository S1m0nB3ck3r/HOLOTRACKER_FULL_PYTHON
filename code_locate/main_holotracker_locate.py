import ttkbootstrap as tb
from ui import HoloTrackerApp
from core import HoloTrackerCore
from controller_threaded import HoloTrackerController
from ui_styles import apply_custom_styles

root = tb.Window(themename="superhero")  # Modern dark theme

# Apply custom styles for rounded corners and enhanced appearance
apply_custom_styles(root)

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
