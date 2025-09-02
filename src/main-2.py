import tkinter as tk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap.scrolled import ScrolledText
import logging
import threading
import queue
import oci
import configparser
import os
import json
import markdown
from tkhtmlview import HTMLScrolledText

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

# AI module
class AI:
    def __init__(self, parent, model, endpoint):
        self.parent = parent
        self.model = model
        self.endpoint = endpoint
        logging.info(f"AI module initialized with model: {self.model}, endpoint: {self.endpoint}")

    def update_model(self, model):
        self.model = model
        logging.info(f"AI model updated to: {self.model}")

    def update_endpoint(self, endpoint):
        self.endpoint = endpoint
        logging.info(f"AI endpoint updated to: {self.endpoint}")

# BaseTenancyData module
class BaseTenancyData:
    def __init__(self, parent):
        self.parent = parent
        self.data = {}
        logging.info("BaseTenancyData module initialized")

    def load(self, mode, param=None):
        self.data = {}
        if mode == "Instance Principal":
            try:
                signer = oci.auth.signers.InstancePrincipalsSecurityTokenSigner()
                client = oci.identity.IdentityClient(config={}, signer=signer)
                tenancy = client.get_tenancy(tenancy_id=signer.tenancy_id).data
                self.data = {"tenancy": {"name": tenancy.name, "id": tenancy.id}}
                logging.info("Loaded tenancy data using Instance Principal")
            except Exception as e:
                logging.error(f"Error loading Instance Principal: {e}")
        elif mode == "Profile":
            try:
                config = oci.config.from_file(profile_name=param)
                client = oci.identity.IdentityClient(config)
                tenancy = client.get_tenancy(tenancy_id=config['tenancy']).data
                self.data = {"tenancy": {"name": tenancy.name, "id": tenancy.id}}
                logging.info(f"Loaded tenancy data using Profile: {param}")
            except Exception as e:
                logging.error(f"Error loading Profile {param}: {e}")
        elif mode == "Cache":
            try:
                cache_dir = 'cache'
                if not os.path.exists(cache_dir):
                    os.makedirs(cache_dir)
                with open(os.path.join(cache_dir, param), 'r') as f:
                    self.data = json.load(f)
                logging.info(f"Loaded tenancy data from Cache: {param}")
            except Exception as e:
                logging.error(f"Error loading Cache {param}: {e}")

# Main application class
class App(ttk.Window):
    def __init__(self):
        super().__init__(themename='litera')  # Light theme for visibility
        self.title("OCI Management App")
        self.geometry("800x600")

        # Queue for long-running tasks
        self.queue = queue.Queue()
        self.running = True  # Control flag for thread

        # Variables
        self.ai_model_var = tk.StringVar(value="default_model")
        self.ai_endpoint_var = tk.StringVar(value="https://api.example.com")
        self.load_from_var = tk.StringVar(value="Instance Principal")
        self.secondary_var = tk.StringVar()
        self.show_console_var = tk.BooleanVar(value=False)
        self.view_mode_var = tk.DoubleVar(value=0.0)  # Slider: 0=Tree, 1=Table

        # Initialize modules
        self.ai = AI(self, self.ai_model_var.get(), self.ai_endpoint_var.get())
        self.base_tenancy_data = BaseTenancyData(self)

        # Configure grid
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(3, weight=0)
        self.grid_columnconfigure(0, weight=1)

        # Styles for visibility
        style = ttk.Style()
        style.configure('TLabel', font=('Helvetica', 10), foreground='black', background='white')
        style.configure('TButton', font=('Helvetica', 10), foreground='black', background='lightgray')
        style.configure('TCheckbutton', font=('Helvetica', 10), foreground='black', background='white')
        style.configure('TCombobox', font=('Helvetica', 10), foreground='black', background='white')
        style.configure('TEntry', font=('Helvetica', 10), foreground='black', background='white')

        # Top frame: Options
        self.options_frame = ttk.Frame(self, padding=10)
        self.options_frame.grid(row=0, column=0, sticky='ew')

        ttk.Label(self.options_frame, text="AI Model:", style='TLabel').pack(side=LEFT, padx=5)
        self.ai_model_entry = ttk.Entry(self.options_frame, textvariable=self.ai_model_var, style='TEntry')
        self.ai_model_entry.pack(side=LEFT, padx=5)
        self.ai_model_entry.bind("<KeyRelease>", self.update_ai_model)

        ttk.Label(self.options_frame, text="AI Endpoint:", style='TLabel').pack(side=LEFT, padx=5)
        self.ai_endpoint_entry = ttk.Entry(self.options_frame, textvariable=self.ai_endpoint_var, style='TEntry')
        self.ai_endpoint_entry.pack(side=LEFT, padx=5)
        self.ai_endpoint_entry.bind("<KeyRelease>", self.update_ai_endpoint)

        ttk.Label(self.options_frame, text="Load From:", style='TLabel').pack(side=LEFT, padx=5)
        self.load_from_combo = ttk.Combobox(
            self.options_frame,
            textvariable=self.load_from_var,
            values=["Instance Principal", "Profile", "Cache"],
            state='readonly',
            style='TCombobox'
        )
        self.load_from_combo.pack(side=LEFT, padx=5)
        self.load_from_combo.bind('<<ComboboxSelected>>', self.update_load_from)

        self.secondary_combo = ttk.Combobox(
            self.options_frame,
            textvariable=self.secondary_var,
            state='disabled',
            style='TCombobox'
        )
        self.secondary_combo.pack(side=LEFT, padx=5)
        self.secondary_combo.bind('<<ComboboxSelected>>', self.load_data)

        self.load_button = ttk.Button(
            self.options_frame,
            text="Load",
            command=self.load_data,
            style='TButton'
        )
        self.load_button.pack(side=LEFT, padx=5)

        self.console_toggle = ttk.Checkbutton(
            self.options_frame,
            text="Show Console Log",
            variable=self.show_console_var,
            command=self.toggle_console,
            style='TCheckbutton'
        )
        self.console_toggle.pack(side=RIGHT)

        # Detail pane: Notebook
        self.detail_notebook = ttk.Notebook(self)
        self.detail_notebook.grid(row=1, column=0, sticky='nsew')

        self.json_data_frame = ttk.Frame(self.detail_notebook)
        self.detail_notebook.add(self.json_data_frame, text="JSON Data")

        # View mode slider
        view_control_frame = ttk.Frame(self.json_data_frame)
        view_control_frame.pack(fill=X, pady=5)
        ttk.Label(view_control_frame, text="Tree", style='TLabel').pack(side=LEFT, padx=5)
        self.view_mode_scale = ttk.Scale(
            view_control_frame,
            from_=0.0,
            to=1.0,
            orient=HORIZONTAL,
            variable=self.view_mode_var,
            command=self.update_view
        )
        self.view_mode_scale.pack(side=LEFT, fill=X, expand=True, padx=5)
        ttk.Label(view_control_frame, text="Table", style='TLabel').pack(side=LEFT, padx=5)

        # Treeview/Table
        self.treeview = ttk.Treeview(self.json_data_frame, show='tree')
        self.treeview.pack(expand=True, fill=BOTH)
        self.tableview = ttk.Treeview(self.json_data_frame, columns=('Key', 'Value'), show='headings')
        self.tableview.heading('Key', text='Key')
        self.tableview.heading('Value', text='Value')
        self.tableview.pack(expand=True, fill=BOTH)
        self.tableview.pack_forget()  # Initially show tree

        # AI Insights: Scrollable Markdown text
        self.ai_insights_frame = ttk.Frame(self, padding=10)
        self.ai_insights_frame.grid(row=2, column=0, sticky='ew')
        ttk.Label(self.ai_insights_frame, text="AI Insights Panel", style='TLabel').pack(fill=X)
        markdown_text = "# AI Insights\n**Model**: {}\n**Endpoint**: {}\nSample *italic* text.".format(
            self.ai_model_var.get(), self.ai_endpoint_var.get()
        )
        html_content = markdown.markdown(markdown_text)
        self.ai_insights_text = HTMLScrolledText(self.ai_insights_frame, height=10)
        self.ai_insights_text.pack(fill=BOTH, expand=True)
        self.ai_insights_text.set_html(html_content)

        # Console Log Frame
        self.console_frame = ttk.Frame(self, padding=5)
        self.log_level_var = tk.StringVar(value='INFO')
        self.log_level_combo = ttk.Combobox(
            self.console_frame,
            textvariable=self.log_level_var,
            values=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
            state='readonly',
            style='TCombobox'
        )
        self.log_level_combo.pack(side=LEFT, padx=5)
        self.log_level_combo.bind('<<ComboboxSelected>>', self.change_log_level)

        self.clear_button = ttk.Button(
            self.console_frame,
            text="Clear",
            command=self.clear_console,
            style='TButton'
        )
        self.clear_button.pack(side=LEFT, padx=5)

        self.console_text = ScrolledText(self.console_frame, height=10, autohide=True, font=('Helvetica', 10))
        self.console_text.pack(side=LEFT, fill=BOTH, expand=True)
        self.console_text.text.configure(background='white', foreground='black')

        self.text_handler = TextHandler(self.console_text)
        logging.getLogger().addHandler(self.text_handler)

        # Initial setup (populate dropdowns but don't load data)
        self.update_load_from(no_load=True)

        # Start background thread after UI initialization
        self.after(100, self.start_background_thread)

    def start_background_thread(self):
        self.processing_thread = threading.Thread(target=self.process_queue, daemon=True)
        self.processing_thread.start()

    def update_ai_model(self, event=None):
        self.ai.update_model(self.ai_model_var.get())
        self.update_ai_insights()

    def update_ai_endpoint(self, event=None):
        self.ai.update_endpoint(self.ai_endpoint_var.get())
        self.update_ai_insights()

    def update_ai_insights(self):
        markdown_text = "# AI Insights\n**Model**: {}\n**Endpoint**: {}\nSample *italic* text.".format(
            self.ai_model_var.get(), self.ai_endpoint_var.get()
        )
        html_content = markdown.markdown(markdown_text)
        self.ai_insights_text.set_html(html_content)

    def update_load_from(self, event=None, no_load=False):
        load_from = self.load_from_var.get()
        self.secondary_combo['state'] = 'disabled'
        self.secondary_var.set('')
        values = []
        if load_from == "Profile":
            config_file = os.path.expanduser('~/.oci/config')
            if os.path.exists(config_file):
                config = configparser.ConfigParser()
                config.read(config_file)
                values = [section for section in config.sections() if section != 'DEFAULT']
        elif load_from == "Cache":
            cache_dir = 'cache'
            if os.path.exists(cache_dir):
                values = [f for f in os.listdir(cache_dir) if f.endswith('.json')]
        self.secondary_combo['values'] = values
        if values:
            self.secondary_var.set(values[0])
            self.secondary_combo['state'] = 'readonly'
        if not no_load:
            self.load_data()

    def load_data(self, event=None):
        mode = self.load_from_var.get()
        param = self.secondary_var.get() if self.secondary_var.get() else None
        self.base_tenancy_data.load(mode, param)
        self.update_view()

    def update_view(self, event=None):
        self.treeview.delete(*self.treeview.get_children())
        self.tableview.delete(*self.tableview.get_children())
        data = self.base_tenancy_data.data
        if self.view_mode_var.get() < 0.5:  # Tree view
            self.tableview.pack_forget()
            self.treeview.pack(expand=True, fill=BOTH)
            self.populate_tree(self.treeview, '', data)
        else:  # Table view
            self.treeview.pack_forget()
            self.tableview.pack(expand=True, fill=BOTH)
            flat_data = self.flatten_dict(data)
            for k, v in flat_data.items():
                self.tableview.insert('', 'end', values=(k, str(v)))

    def populate_tree(self, tree, parent, data):
        if isinstance(data, dict):
            for key, value in data.items():
                node = tree.insert(parent, 'end', text=key)
                self.populate_tree(tree, node, value)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                node = tree.insert(parent, 'end', text=f"[{i}]")
                self.populate_tree(tree, node, item)
        else:
            tree.insert(parent, 'end', text=str(data))

    def flatten_dict(self, d, parent_key='', sep='.'):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self.flatten_dict(v, new_key, sep).items())
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
        while self.running:
            try:
                task = self.queue.get_nowait()  # Non-blocking
                logging.info(f"Processing queue task: {task}")
                self.queue.task_done()
            except queue.Empty:
                self.after(100, self.process_queue)  # Schedule next check
                break
            except Exception as e:
                logging.error(f"Error in queue processing: {e}")

    def destroy(self):
        self.running = False  # Stop background thread
        super().destroy()

if __name__ == "__main__":
    app = App()
    app.mainloop()