import tkinter as tk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap.scrolled import ScrolledText
import logging
import threading
import queue
import oci  # Oracle Cloud Infrastructure SDK for tenancy data interactions

# Configure logging with thread name and module name
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(module)s - %(levelname)s - %(message)s'
)

# Custom logging handler for Text widget
class TextHandler(logging.Handler):
    def __init__(self, text_widget):
        super().__init__()
        self.text = text_widget
        self.setFormatter(logging.Formatter('%(asctime)s - %(threadName)s - %(module)s - %(levelname)s - %(message)s'))

    def emit(self, record):
        msg = self.format(record)
        self.text.insert(tk.END, msg + '\n')
        self.text.see(tk.END)

# Placeholder for AI module (to be implemented)
class AI:
    def __init__(self, parent):
        self.parent = parent
        logging.info("AI module initialized")

# Placeholder for TenancyData module (to be implemented, will use OCI SDK)
class TenancyData:
    def __init__(self, parent):
        self.parent = parent
        self.config = oci.config.from_file()  # Load OCI config for tenancy interactions
        logging.info("TenancyData module initialized")

# Main application class
class App(ttk.Window):
    def __init__(self):
        super().__init__(themename='litera')  # Use litera theme for better visibility
        self.title("OCI Management App")
        self.geometry("800x600")
        
        # Queue for long-running activities (e.g., OCI API calls in threads)
        self.queue = queue.Queue()
        
        # Start a background thread for processing queue (threaded actions)
        self.processing_thread = threading.Thread(target=self.process_queue, daemon=True)
        self.processing_thread.start()
        
        logging.info("Main App initialized")
        
        # Initialize modules
        self.ai = AI(self)
        self.tenancy_data = TenancyData(self)
        
        # Configure grid for responsive layout
        self.grid_rowconfigure(1, weight=1)  # Detail pane expands
        self.grid_rowconfigure(3, weight=0)  # Console frame does not expand by default
        self.grid_columnconfigure(0, weight=1)
        
        # Configure styles for better visibility
        style = ttk.Style()
        style.configure('TLabel', font=('Helvetica', 10), foreground='black', background='white')
        style.configure('TButton', font=('Helvetica', 10), foreground='black', background='lightgray')
        style.configure('TCheckbutton', font=('Helvetica', 10), foreground='black', background='white')
        style.configure('TCombobox', font=('Helvetica', 10), foreground='black', background='white')
        
        # Top frame: Options (across the top)
        self.options_frame = ttk.Frame(self, padding=10)
        self.options_frame.grid(row=0, column=0, sticky='ew')
        ttk.Label(self.options_frame, text="Options Panel", bootstyle=INFO, style='TLabel').pack(side=LEFT, fill=X, expand=True)
        
        # Toggle for console log
        self.show_console_var = tk.BooleanVar(value=False)
        self.console_toggle = ttk.Checkbutton(
            self.options_frame,
            text="Show Console Log",
            variable=self.show_console_var,
            command=self.toggle_console,
            bootstyle=INFO,
            style='TCheckbutton'
        )
        self.console_toggle.pack(side=RIGHT)
        
        # Middle frame: Detail pane
        self.detail_pane = ttk.Frame(self, padding=10)
        self.detail_pane.grid(row=1, column=0, sticky='nsew')
        ttk.Label(self.detail_pane, text="Detail Pane", bootstyle=WARNING, style='TLabel').pack(expand=True, fill=BOTH)
        
        # Bottom frame: AI Insights (across the bottom)
        self.ai_insights_frame = ttk.Frame(self, padding=10)
        self.ai_insights_frame.grid(row=2, column=0, sticky='ew')
        ttk.Label(self.ai_insights_frame, text="AI Insights Panel", bootstyle=SUCCESS, style='TLabel').pack(fill=X)
        
        # 4th frame: Console log (very bottom, toggleable)
        self.console_frame = ttk.Frame(self, padding=5)
        # Initially hidden
        # self.console_frame.grid(row=3, column=0, sticky='ew')  # Will be gridded in toggle
        
        # Log level selector
        self.log_level_var = tk.StringVar(value='INFO')
        self.log_level_combo = ttk.Combobox(
            self.console_frame,
            textvariable=self.log_level_var,
            values=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
            state='readonly',
            bootstyle=SECONDARY,
            style='TCombobox'
        )
        self.log_level_combo.pack(side=LEFT, padx=5)
        self.log_level_combo.bind('<<ComboboxSelected>>', self.change_log_level)
        
        # Clear button
        self.clear_button = ttk.Button(
            self.console_frame,
            text="Clear",
            command=self.clear_console,
            bootstyle=SECONDARY,
            style='TButton'
        )
        self.clear_button.pack(side=LEFT, padx=5)
        
        # Scrolling text area for logs
        self.console_text = ScrolledText(self.console_frame, height=10, autohide=True, font=('Helvetica', 10))
        self.console_text.pack(side=LEFT, fill=BOTH, expand=True)
        # Explicitly set text area colors for visibility
        self.console_text.text.configure(background='white', foreground='black')
        
        # Add custom handler to logger
        self.text_handler = TextHandler(self.console_text)
        logging.getLogger().addHandler(self.text_handler)
    
    def toggle_console(self):
        if self.show_console_var.get():
            self.console_frame.grid(row=3, column=0, sticky='ew')
        else:
            self.console_frame.grid_remove()
    
    def change_log_level(self, event=None):
        level = self.log_level_var.get()
        logging.getLogger().setLevel(getattr(logging, level))
        logging.info(f"Log level changed to {level}")
    
    def clear_console(self):
        self.console_text.delete(1.0, tk.END)
        logging.info("Console log cleared")
    
    def process_queue(self):
        """Background thread to handle long-running tasks via queue (e.g., OCI API calls)."""
        while True:
            try:
                task = self.queue.get()
                # Placeholder for task processing (e.g., self.tenancy_data.some_oci_action(task))
                logging.info(f"Processing queue task: {task}")
                self.queue.task_done()
            except Exception as e:
                logging.error(f"Error in queue processing: {e}")

if __name__ == "__main__":
    app = App()
    app.mainloop()
