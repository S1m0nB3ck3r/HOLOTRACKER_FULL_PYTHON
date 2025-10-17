"""
UI Styles and Theme Configuration for HoloTracker
Provides enhanced visual styling with rounded corners and modern aesthetics
"""

import ttkbootstrap as tb
from ttkbootstrap.constants import *

# Unicode icons for buttons (platform-independent)
ICONS = {
    # File operations
    'folder': '📁',
    'browse': '🔍',
    'file': '📄',
    'save': '💾',
    'load': '📂',
    
    # Actions
    'compute': '⚙️',
    'process': '▶️',
    'start': '▶️',
    'stop': '⏹️',
    'pause': '⏸️',
    'cancel': '❌',
    'refresh': '🔄',
    
    # Test mode
    'enter': '🚀',
    'exit': '🚪',
    'test': '🧪',
    
    # Zoom/View
    'zoom_in': '🔍+',
    'zoom_out': '🔍-',
    'reset': '↺',
    'stretch': '↔️',
    
    # Info
    'help': '❓',
    'info': 'ℹ️',
    'warning': '⚠️',
    'error': '⚠️',
    'success': '✅',
    
    # Navigation
    'next': '→',
    'previous': '←',
    'up': '↑',
    'down': '↓',
    
    # Analysis
    'analyze': '📊',
    'chart': '📈',
    'stats': '📊',
    'slices': '🔬',
    
    # Batch
    'batch': '📦',
    'list': '📋',
}

def apply_custom_styles(root):
    """
    Apply custom styles to the application for enhanced visual appearance.
    This includes rounded corners and improved spacing.
    
    Args:
        root: The ttkbootstrap Window instance
    """
    style = tb.Style()
    
    # ==================== BUTTONS ====================
    # Primary button style (rounded, with emphasis)
    style.configure(
        'Primary.TButton',
        font=('Segoe UI', 10, 'bold'),
        borderwidth=0,
        focuscolor='none',
        padding=(20, 10),
    )
    
    # Success button style (green, for positive actions)
    style.configure(
        'Success.TButton',
        font=('Segoe UI', 10, 'bold'),
        borderwidth=0,
        padding=(20, 10),
    )
    
    # Danger button style (red, for destructive actions)
    style.configure(
        'Danger.TButton',
        font=('Segoe UI', 10, 'bold'),
        borderwidth=0,
        padding=(20, 10),
    )
    
    # Info button style (blue, for informational actions)
    style.configure(
        'Info.TButton',
        font=('Segoe UI', 10, 'bold'),
        borderwidth=0,
        padding=(20, 10),
    )
    
    # Secondary button style (smaller, less emphasis)
    style.configure(
        'Secondary.TButton',
        font=('Segoe UI', 9),
        borderwidth=0,
        padding=(15, 8),
    )
    
    # Icon button style (compact with icon)
    style.configure(
        'Icon.TButton',
        font=('Segoe UI', 10),
        borderwidth=0,
        padding=(10, 8),
    )
    
    # ==================== LABELS ====================
    # Title label style
    style.configure(
        'Title.TLabel',
        font=('Segoe UI', 12, 'bold'),
        padding=5,
    )
    
    # Subtitle label style
    style.configure(
        'Subtitle.TLabel',
        font=('Segoe UI', 10, 'bold'),
        padding=3,
    )
    
    # Status label style
    style.configure(
        'Status.TLabel',
        font=('Consolas', 9),
        padding=5,
    )
    
    # ==================== FRAMES ====================
    # Card frame style (with subtle border)
    style.configure(
        'Card.TFrame',
        borderwidth=1,
        relief='solid',
        padding=10,
    )
    
    # ==================== ENTRIES ====================
    # Enhanced entry style
    style.configure(
        'TEntry',
        borderwidth=1,
        relief='solid',
        padding=5,
    )
    
    # ==================== LABELFRAMES ====================
    # Enhanced labelframe style
    style.configure(
        'TLabelframe',
        borderwidth=2,
        relief='groove',
        padding=10,
    )
    
    style.configure(
        'TLabelframe.Label',
        font=('Segoe UI', 10, 'bold'),
    )
    
    # ==================== NOTEBOOKS (TABS) ====================
    style.configure(
        'TNotebook',
        borderwidth=0,
        padding=5,
    )
    
    style.configure(
        'TNotebook.Tab',
        padding=(20, 10),
        font=('Segoe UI', 10),
    )

def get_icon(icon_name):
    """
    Get an icon from the ICONS dictionary.
    
    Args:
        icon_name: Name of the icon
        
    Returns:
        Unicode character for the icon, or empty string if not found
    """
    return ICONS.get(icon_name, '')

def format_button_text(text, icon_name=None):
    """
    Format button text with optional icon.
    
    Args:
        text: Button text
        icon_name: Optional icon name from ICONS dictionary
        
    Returns:
        Formatted text with icon
    """
    if icon_name and icon_name in ICONS:
        icon = ICONS[icon_name]
        return f"{icon}  {text}"
    return text

# Bootstyle presets for common button types
BUTTON_STYLES = {
    'primary': 'primary',
    'secondary': 'secondary',
    'success': 'success',
    'danger': 'danger',
    'warning': 'warning',
    'info': 'info',
    'light': 'light',
    'dark': 'dark',
}

# Color scheme enhancements (works with superhero theme)
COLOR_PALETTE = {
    'background': '#1e1e1e',
    'surface': '#2a2a2a',
    'primary': '#4e73df',
    'success': '#1cc88a',
    'danger': '#e74a3b',
    'warning': '#f6c23e',
    'info': '#36b9cc',
    'text': '#ffffff',
    'text_secondary': '#858796',
}

def create_icon_button(parent, text, icon_name, command, bootstyle='primary', width=None):
    """
    Create a button with icon and text.
    
    Args:
        parent: Parent widget
        text: Button text
        icon_name: Icon name from ICONS dictionary
        command: Button command callback
        bootstyle: Bootstrap style (primary, success, danger, etc.)
        width: Optional button width
        
    Returns:
        ttk.Button instance
    """
    button_text = format_button_text(text, icon_name)
    if width:
        return tb.Button(
            parent,
            text=button_text,
            command=command,
            bootstyle=bootstyle,
            width=width
        )
    else:
        return tb.Button(
            parent,
            text=button_text,
            command=command,
            bootstyle=bootstyle
        )
