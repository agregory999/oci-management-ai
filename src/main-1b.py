import tkinter as tk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap.scrolled import ScrolledText
import logging
import threading
import queue
import oci  # Oracle Cloud Infrastructure SDK for tenancy data interactions
from oci.auth.signers import InstancePrincipalsSecurityTokenSigner
import configparser
import os
from os.path import expanduser
import json
# pip install markdown tkhtmlview  # For AI insights Markdown rendering
try:
    import markdown
    from tkhtmlview import HTMLScrolledText
except ImportError:
    # Fallback if not installed
    HTMLScrolledText = ScrolledText
    markdown = lambda text: text  # No conversion

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
        self.model = parent.ai_model_var.get()
        self.endpoint = parent.ai_endpoint_var.get()
        logging.info(f"AI module initialized with model: {self.model}, endpoint: {self.endpoint}")

# Placeholder for BaseTenancyData module (to be implemented, will use OCI SDK)
class BaseTenancyData:
    def __init__(self, parent):
        self.parent = parent
        self.data = {}  # Placeholder for tenancy data
        self.config = None
        self.signer = None
        self.client = None
        logging.info("BaseTenancyData module initialized")

    def load(self, mode, param=None):
        if mode == "Instance Principal":
            try:
                self.signer = InstancePrincipalsSecurityTokenSigner()
                self.client = oci.identity.IdentityClient(config={}, signer=self.signer)
                logging.info("Loaded using Instance Principal")
            except Exception as e:
                logging.error(f"Error loading Instance Principal: {e}")
                return
        elif mode == "Profile":
            try:
                self.config = oci.config.from_file(profile_name=param)
                self.client = oci.identity.IdentityClient(self.config)
                logging.info(f"Loaded using profile: {param}")
            except Exception as e:
                logging.error(f"Error loading profile {param}: {e}")
                return
        elif mode == "Cache":
            cache_dir = 'cache'
            file_path = os.path.join(cache_dir, param)
            try:
                with open(file_path, 'r') as f:
                    self.data = json.load(f)
                logging.info(f"Loaded from cache file: {param}")
                return  # No further fetch for cache
            except Exception as e:
                logging.error(f"Error loading cache {param}: {e}")
                return

        # Placeholder for fetching data using client
        if self.client:
            try:
                # Example: Get tenancy details
                tenancy = self.client.get_tenancy(tenancy_id=self.config.get('tenancy') if self.config else self.signer.tenancy_id).data
                self.data = {"tenancy_name": tenancy.name, "tenancy_id": tenancy.id}  # Placeholder data
                logging.info("Fetched tenancy data")
            except Exception as e:
                logging.error(f"Error fetching data: {e}")

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
        
        # Variables for AI
        self.ai_model_var = tk.StringVar(value="default_model")
        self.ai_endpoint_var = tk.StringVar(value="default_endpoint")
        
        # Initialize modules
        self.ai = AI(self)
        self.base_tenancy_data = BaseTenancyData(self)
        
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
        style.configure('TEntry', font=('Helvetica', 10), foreground='black', background='white')
        
        # Top frame: Options (across the top)
        self.options_frame = ttk.Frame(self, padding=10)
        self.options_frame.grid(row=0, column=0, sticky='ew')
        
        # AI Model entry
        ttk.Label(self.options_frame, text="AI Model:", style='TLabel').pack(side=LEFT, padx=5)
        self.ai_model_entry = ttk.Entry(self.options_frame, textvariable=self.ai_model_var, bootstyle=PRIMARY, style='TEntry')
        self.ai_model_entry.pack(side=LEFT, padx=5)
        
        # AI Endpoint entry
        ttk.Label(self.options_frame, text="AI Endpoint:", style='TLabel').pack(side=LEFT, padx=5)
        self.ai_endpoint_entry = ttk.Entry(self.options_frame, textvariable=self.ai_endpoint_var, bootstyle=PRIMARY, style='TEntry')
        self.ai_endpoint_entry.pack(side=LEFT, padx=5)
        
        # Load from dropdown
        ttk.Label(self.options_frame, text="Load From:", style='TLabel').pack(side=LEFT, padx=5)
        self.load_from_var = tk.StringVar(value="Instance Principal")
        self.load_from_combo = ttk.Combobox(
            self.options_frame,
            textvariable=self.load_from_var,
            values=["Instance Principal", "Profile", "Cache"],
            state='readonly',
            bootstyle=SECONDARY,
            style='TCombobox'
        )
        self.load_from_combo.pack(side=LEFT, padx=5)
        self.load_from_var.trace('w', self.on_load_from_change)
        
        # Secondary dropdown (for profiles or cache files)
        self.secondary_var = tk.StringVar()
        self.secondary_combo = ttk.Combobox(
            self.options_frame,
            textvariable=self.secondary_var,
            state='disabled',
            bootstyle=SECONDARY,
            style='TCombobox'
        )
        self.secondary_combo.pack(side=LEFT, padx=5)
        
        # Load button
        self.load_button = ttk.Button(
            self.options_frame,
            text="Load",
            command=self.load_data,
            bootstyle=PRIMARY,
            style='TButton'
        )
        self.load_button.pack(side=LEFT, padx=5)
        
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
        
        # Middle frame: Detail pane (Notebook)
        self.detail_pane = ttk.Notebook(self)
        self.detail_pane.grid(row=1, column=0, sticky='nsew')
        
        # JSON Data tab
        self.json_tab = ttk.Frame(self.detail_pane)
        self.detail_pane.add(self.json_tab, text="JSON Data")
        
        # View slider in JSON tab
        ttk.Label(self.json_tab, text="Tree View", style='TLabel').pack(side=LEFT, padx=5)
        self.view_var = tk.IntVar(value=0)
        self.view_scale = ttk.Scale(
            self.json_tab,
            from_=0,
            to=1,
            orient='horizontal',
            variable=self.view_var
        )
        self.view_scale.pack(side=LEFT, fill=X, expand=True, padx=5)
        ttk.Label(self.json_tab, text="Table View", style='TLabel').pack(side=LEFT, padx=5)
        self.view_var.trace('w', self.update_view)
        
        # View frame for tree/table
        self.view_frame = ttk.Frame(self.json_tab)
        self.view_frame.pack(expand=True, fill=BOTH, padx=5, pady=5)
        self.view_frame.grid_rowconfigure(0, weight=1)
        self.view_frame.grid_columnconfigure(0, weight=1)
        
        # Tree view
        self.tree_view = ttk.Treeview(self.view_frame)
        self.tree_view.grid(row=0, column=0, sticky='nsew')
        
        # Table view
        self.table_view = ttk.Treeview(self.view_frame, columns=('Key', 'Value'), show='headings')
        self.table_view.heading('Key', text='Key')
        self.table_view.heading('Value', text='Value')
        self.table_view.grid(row=0, column=0, sticky='nsew')
        self.table_view.grid_remove()  # Initially show tree
        
        # Bottom frame: AI Insights (across the bottom)
        self.ai_insights_frame = ttk.Frame(self, padding=10)
        self.ai_insights_frame.grid(row=2, column=0, sticky='ew')
        
        # Scrollable text with Markdown HTML
        markdown_text = "# AI Insights\nPlaceholder content with **bold** and *italic*."  # Placeholder Markdown
        html_content = markdown.markdown(markdown_text)
        self.ai_insights_text = HTMLScrolledText(self.ai_insights_frame, html=html_content, height=10)
        self.ai_insights_text.pack(fill=BOTH, expand=True)
        
        # 4th frame: Console log (very bottom, toggleable)
        self.console_frame = ttk.Frame(self, padding=5)
        # Initially hidden
        
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
        self.console_text.text.configure(background='white', foreground='black')
        
        # Add custom handler to logger
        self.text_handler = TextHandler(self.console_text)
        logging.getLogger().addHandler(self.text_handler)
        
        # Initial update
        self.on_load_from_change()
        self.update_view()
    
    def on_load_from_change(self, *args):
        choice = self.load_from_var.get()
        self.secondary_combo['values'] = []
        self.secondary_var.set('')
        self.secondary_combo['state'] = 'disabled'
        if choice == "Profile":
            profiles = self.get_profiles()
            self.secondary_combo['values'] = profiles
            if profiles:
                self.secondary_var.set(profiles[0])
                self.secondary_combo['state'] = 'readonly'
        elif choice == "Cache":
            cache_files = self.get_cache_files()
            self.secondary_combo['values'] = cache_files
            if cache_files:
                self.secondary_var.set(cache_files[0])
                self.secondary_combo['state'] = 'readonly'
    
    def get_profiles(self):
        config_file = expanduser("~/.oci/config")
        if os.path.exists(config_file):
            config = configparser.ConfigParser()
            config.read(config_file)
            return config.sections()
        return []
    
    def get_cache_files(self):
        cache_dir = 'cache'
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        return [f for f in os.listdir(cache_dir) if f.endswith('.json')]
    
    def load_data(self):
        choice = self.load_from_var.get()
        param = self.secondary_var.get() if choice in ["Profile", "Cache"] else None
        self.base_tenancy_data.load(choice, param)
        self.update_view()  # Refresh view after loading data
        # Update AI if needed
        self.ai.model = self.ai_model_var.get()
        self.ai.endpoint = self.ai_endpoint_var.get()
        logging.info(f"AI updated with model: {self.ai.model}, endpoint: {self.ai.endpoint}")
    
    def update_view(self, *args):
        view = self.view_var.get()
        data = self.base_tenancy_data.data
        if view == 0:  # Tree View
            self.table_view.grid_remove()
            self.tree_view.grid(row=0, column=0, sticky='nsew')
            self.tree_view.delete(*self.tree_view.get_children())
            self.insert_tree(self.tree_view, '', data)
        else:  # Table View
            self.tree_view.grid_remove()
            self.table_view.grid(row=0, column=0, sticky='nsew')
            self.table_view.delete(*self.table_view.get_children())
            flat_data = self.flatten_dict(data)
            for k, v in flat_data.items():
                self.table_view.insert('', 'end', values=(k, str(v)))
    
    def insert_tree(self, tree, parent, data):
        if isinstance(data, dict):
            for k, v in data.items():
                child = tree.insert(parent, 'end', text=k)
                self.insert_tree(tree, child, v)
        elif isinstance(data, list):
            for i, v in enumerate(data):
                child = tree.insert(parent, 'end', text=f"[{i}]")
                self.insert_tree(tree, child, v)
        else:
            tree.insert(parent, 'end', text=str(data))
    
    def flatten_dict(self, d, parent_key='', sep='_'):
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, dict):
                items.extend(self.flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
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
                # Placeholder for task processing (e.g., self.base_tenancy_data.some_oci_action(task))
                logging.info(f"Processing queue task: {task}")
                self.queue.task_done()
            except Exception as e:
                logging.error(f"Error in queue processing: {e}")

if __name__ == "__main__":
    app = App()
    app.mainloop()