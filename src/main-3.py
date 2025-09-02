# Stubbed out AI call and working logging
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
from datetime import datetime, timezone

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
    def __init__(self, parent, model, endpoint, config=None, signer=None):
        self.parent = parent
        self.model = model
        self.endpoint = endpoint
        self.config = config
        self.signer = signer
        self.client = None
        self.compartment_id = None
        self.cache = []
        self.cache_file = 'cache/ai_cache.json'
        # Load existing cache
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    self.cache = json.load(f)
            except Exception as e:
                logging.error(f"Failed to load AI cache: {e}")
        # Initialize OCI client
        self.initialize_client()
        logging.info(f"AI module initialized with model: {self.model}, endpoint: {self.endpoint}")

    def initialize_client(self):
        try:
            if self.signer:
                self.client = oci.generative_ai_inference.GenerativeAiInferenceClient(
                    config={}, signer=self.signer, service_endpoint=self.endpoint
                )
            elif self.config:
                self.client = oci.generative_ai_inference.GenerativeAiInferenceClient(
                    config=self.config, service_endpoint=self.endpoint
                )
            # Get compartment ID from tenancy data or config
            if self.parent.base_tenancy_data.data.get('tenancy', {}).get('id'):
                self.compartment_id = self.parent.base_tenancy_data.data['tenancy']['id']
            elif self.config and 'tenancy' in self.config:
                self.compartment_id = self.config['tenancy']
            else:
                self.compartment_id = 'default-compartment-id'  # Placeholder
                logging.warning("No compartment ID found; using placeholder")
        except Exception as e:
            logging.error(f"Failed to initialize AI client: {e}")
            self.client = None

    def save_cache(self, cache):
        try:
            cache_dir = 'cache'
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            with open(self.cache_file, 'w') as f:
                json.dump(cache, f, indent=2)
            logging.info(f"Saved AI cache to {self.cache_file}")
        except Exception as e:
            logging.error(f"Failed to save AI cache: {e}")

    def update_model(self, model):
        self.model = model
        logging.info(f"AI model updated to: {self.model}")

    def update_endpoint(self, endpoint):
        self.endpoint = endpoint
        self.initialize_client()
        logging.info(f"AI endpoint updated to: {self.endpoint}")

    def query_genai(self, prompt, cache_type=None, cache_query=None, cache=None, **kwargs):
        """
        Query OCI Generative AI with a prompt, optionally caching the result.

        Args:
            prompt: The prompt string to send to the API (markdown output requested).
            cache_type: Type of query for caching (e.g., 'analyze_policy_statement').
            cache_query: Query string for cache lookup (if None, no caching).
            cache: List to store cache entries (if None, no caching).
            **kwargs: Optional API parameters (e.g., max_tokens, temperature).

        Returns:
            Markdown string or error message.
        """
        start_time = datetime.now()
        logging.debug("Querying GenAI with prompt: %s", prompt)

        # Use instance cache if not provided
        cache = cache if cache is not None else self.cache

        # Check cache if provided
        if cache_type and cache_query and cache:
            for entry in cache:
                if entry.get("type") == cache_type and entry.get("query") == cache_query:
                    logging.debug("Cache hit for %s: %s", cache_type, cache_query[:100])
                    return entry.get("result", "Error: Cache entry missing result")

        if not self.client:
            result = "Error: AI client not initialized"
            logging.error(result)
            return result

        # Set default API parameters
        params = {
            "max_tokens": 4096,
            "temperature": 0.7,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "top_p": 0.95,
            "top_k": 50
        }
        params.update(kwargs)

        try:
            # Prepare API request
            chat_detail = oci.generative_ai_inference.models.ChatDetails()
            chat_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(model_id=self.model)
            content = oci.generative_ai_inference.models.TextContent()
            content.text = f"{prompt}\n\nProvide the response in strict markdown format. Avoid empty lines in lists and ensure all content is concise and relevant. Format the policy statement in a code block (```) with no code type specified. Use unordered lists (- item) for descriptions, ensuring each list item has meaningful content and no empty items."
            chat_request = oci.generative_ai_inference.models.GenericChatRequest()
            chat_request.api_format = oci.generative_ai_inference.models.BaseChatRequest.API_FORMAT_GENERIC
            chat_request.messages = [oci.generative_ai_inference.models.Message(role="USER", content=[content])]
            for key, value in params.items():
                setattr(chat_request, key, value)
            chat_detail.chat_request = chat_request
            chat_detail.compartment_id = self.compartment_id

            # Call API
            response = self.client.chat(chat_detail)
            raw_content = response.data.chat_response.choices[0].message.content
            logging.info(f"Raw content type: {type(raw_content)}")
            logging.info(f"Raw content length: {len(raw_content)}")

            # Simplified response handling
            if isinstance(raw_content, list) and len(raw_content) > 0 and isinstance(raw_content[0], oci.generative_ai_inference.models.text_content.TextContent):
                resp_json = json.loads(str(raw_content[0]))
                resp_text = resp_json['text']
                logging.info("Extracted markdown: %s", resp_text[:100])
            else:
                resp_text = f"Error: Unexpected API response format: {type(raw_content)}"
                logging.error(resp_text)

            # Cache result if requested
            if cache_type and cache_query and cache is not None:
                cache.append({
                    "date": datetime.now(timezone.utc).isoformat(),
                    "type": cache_type,
                    "query": cache_query,
                    "result": resp_text
                })
                self.save_cache(cache)

            logging.info("Query completed in %s seconds", (datetime.now() - start_time).total_seconds())
            return resp_text

        except oci.exceptions.ServiceError as e:
            result = f"Error: API call failed ({e.status}): {str(e)}"
            logging.error(result)
            if cache_type and cache_query and cache is not None:
                cache.append({
                    "date": datetime.now(timezone.utc).isoformat(),
                    "type": cache_type,
                    "query": cache_query,
                    "result": result
                })
                self.save_cache(cache)
            return result
        except Exception as e:
            result = f"Error: {str(e)}"
            logging.error(result)
            if cache_type and cache_query and cache is not None:
                cache.append({
                    "date": datetime.now(timezone.utc).isoformat(),
                    "type": cache_type,
                    "query": cache_query,
                    "result": result
                })
                self.save_cache(cache)
            return result

    def test(self, input_text):
        logging.info(f"AI test called with input: {input_text}")
        return self.query_genai(
            prompt=input_text,
            cache_type="test_ai",
            cache_query=input_text,
            cache=self.cache
        )

# BaseTenancyData module
class BaseTenancyData:
    def __init__(self, parent, recursive):
        self.parent = parent
        self.recursive = recursive
        self.data = {}
        logging.info(f"BaseTenancyData module initialized with recursive: {self.recursive}")

    def load(self, mode, param=None):
        self.data = {}
        if mode == "Instance Principal":
            try:
                signer = oci.auth.signers.InstancePrincipalsSecurityTokenSigner()
                client = oci.identity.IdentityClient(config={}, signer=signer)
                tenancy = client.get_tenancy(tenancy_id=signer.tenancy_id).data
                self.data = {"tenancy": {"name": tenancy.name, "id": tenancy.id}}
                logging.info("Loaded tenancy data using Instance Principal")
                return signer
            except Exception as e:
                logging.error(f"Error loading Instance Principal: {e}")
                return None
        elif mode == "Profile":
            try:
                config = oci.config.from_file(profile_name=param)
                client = oci.identity.IdentityClient(config)
                tenancy = client.get_tenancy(tenancy_id=config['tenancy']).data
                self.data = {"tenancy": {"name": tenancy.name, "id": tenancy.id}}
                logging.info(f"Loaded tenancy data using Profile: {param}")
                return config
            except Exception as e:
                logging.error(f"Error loading Profile {param}: {e}")
                return None
        elif mode == "Cache":
            try:
                cache_dir = 'cache'
                if not os.path.exists(cache_dir):
                    os.makedirs(cache_dir)
                with open(os.path.join(cache_dir, param), 'r') as f:
                    self.data = json.load(f)
                logging.info(f"Loaded tenancy data from Cache: {param}")
                return None
            except Exception as e:
                logging.error(f"Error loading Cache {param}: {e}")
                return None

# Main application class
class App(ttk.Window):
    def __init__(self):
        super().__init__(themename='litera')  # Light theme for visibility
        self.title("OCI Management App")
        self.geometry("1200x800")  # Larger window size

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
        self.recursive_var = tk.BooleanVar(value=False)  # Recursive checkbox
        self.test_ai_var = tk.StringVar(value="")  # Test AI input

        # Initialize modules
        self.base_tenancy_data = BaseTenancyData(self, self.recursive_var.get())
        self.ai = AI(self, self.ai_model_var.get(), self.ai_endpoint_var.get())

        # Configure grid
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
        self.grid_columnconfigure(0, weight=1)

        # Styles for visibility
        style = ttk.Style()
        style.configure('TLabel', font=('Helvetica', 10), foreground='black', background='white')
        style.configure('TButton', font=('Helvetica', 10), foreground='black', background='lightgray')
        style.configure('TCheckbutton', font=('Helvetica', 10), foreground='black', background='white')
        style.configure('TCombobox', font=('Helvetica', 10), foreground='black', background='white')
        style.configure('TEntry', font=('Helvetica', 10), foreground='black', background='white')
        style.configure('TLabelFrame.Label', font=('Helvetica', 10, 'bold'), foreground='black', background='white')

        # Main notebook
        self.notebook = ttk.Notebook(self)
        self.notebook.grid(row=0, column=0, sticky='nsew')

        # START tab
        self.start_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.start_frame, text="START")

        # General group
        general_frame = ttk.LabelFrame(self.start_frame, text="General", padding=10)
        general_frame.pack(fill=X, padx=10, pady=5)
        ttk.Label(general_frame, text="Configure general application settings.", font=('Helvetica', 10)).pack(anchor='w', pady=2)
        self.console_toggle = ttk.Checkbutton(
            general_frame,
            text="Show Console Log",
            variable=self.show_console_var,
            command=self.toggle_console,
            style='TCheckbutton'
        )
        self.console_toggle.pack(anchor='w', padx=5, pady=2)
        log_level_frame = ttk.Frame(general_frame)
        log_level_frame.pack(fill=X, pady=2)
        ttk.Label(log_level_frame, text="Log Level:", style='TLabel').pack(side=LEFT, padx=5)
        self.log_level_var = tk.StringVar(value='INFO')
        self.log_level_combo = ttk.Combobox(
            log_level_frame,
            textvariable=self.log_level_var,
            values=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
            state='readonly',
            style='TCombobox'
        )
        self.log_level_combo.pack(side=LEFT, padx=5)
        self.log_level_combo.bind('<<ComboboxSelected>>', self.change_log_level)

        # Tenancy group
        tenancy_frame = ttk.LabelFrame(self.start_frame, text="Tenancy", padding=10)
        tenancy_frame.pack(fill=X, padx=10, pady=5)
        ttk.Label(tenancy_frame, text="Select the tenancy data source and options.", font=('Helvetica', 10)).pack(anchor='w', pady=2)
        recursive_check = ttk.Checkbutton(
            tenancy_frame,
            text="Recursive",
            variable=self.recursive_var,
            command=self.update_recursive,
            style='TCheckbutton'
        )
        recursive_check.pack(anchor='w', padx=5, pady=2)
        load_from_frame = ttk.Frame(tenancy_frame)
        load_from_frame.pack(fill=X, pady=2)
        ttk.Label(load_from_frame, text="Load From:", style='TLabel').pack(side=LEFT, padx=5)
        self.load_from_combo = ttk.Combobox(
            load_from_frame,
            textvariable=self.load_from_var,
            values=["Instance Principal", "Profile", "Cache"],
            state='readonly',
            style='TCombobox'
        )
        self.load_from_combo.pack(side=LEFT, padx=5)
        self.load_from_combo.bind('<<ComboboxSelected>>', self.update_load_from)
        self.secondary_combo = ttk.Combobox(
            load_from_frame,
            textvariable=self.secondary_var,
            state='disabled',
            style='TCombobox'
        )
        self.secondary_combo.pack(side=LEFT, padx=5)
        self.secondary_combo.bind('<<ComboboxSelected>>', self.load_data)
        self.load_button = ttk.Button(
            load_from_frame,
            text="Load",
            command=self.load_data,
            style='TButton'
        )
        self.load_button.pack(side=LEFT, padx=5)

        # AI Config group
        ai_config_frame = ttk.LabelFrame(self.start_frame, text="AI Config", padding=10)
        ai_config_frame.pack(fill=X, padx=10, pady=5)
        ttk.Label(ai_config_frame, text="Configure AI model and endpoint for insights.", font=('Helvetica', 10)).pack(anchor='w', pady=2)
        model_frame = ttk.Frame(ai_config_frame)
        model_frame.pack(fill=X, pady=2)
        ttk.Label(model_frame, text="AI Model:", style='TLabel').pack(side=LEFT, padx=5)
        self.ai_model_entry = ttk.Entry(model_frame, textvariable=self.ai_model_var, style='TEntry')
        self.ai_model_entry.pack(side=LEFT, padx=5, fill=X, expand=True)
        self.ai_model_entry.bind("<KeyRelease>", self.update_ai_model)
        endpoint_frame = ttk.Frame(ai_config_frame)
        endpoint_frame.pack(fill=X, pady=2)
        ttk.Label(endpoint_frame, text="AI Endpoint:", style='TLabel').pack(side=LEFT, padx=5)
        self.ai_endpoint_entry = ttk.Entry(endpoint_frame, textvariable=self.ai_endpoint_var, style='TEntry')
        self.ai_endpoint_entry.pack(side=LEFT, padx=5, fill=X, expand=True)
        self.ai_endpoint_entry.bind("<KeyRelease>", self.update_ai_endpoint)
        test_ai_frame = ttk.Frame(ai_config_frame)
        test_ai_frame.pack(fill=X, pady=2)
        ttk.Label(test_ai_frame, text="Test AI Input:", style='TLabel').pack(side=LEFT, padx=5)
        self.test_ai_entry = ttk.Entry(test_ai_frame, textvariable=self.test_ai_var, style='TEntry')
        self.test_ai_entry.pack(side=LEFT, padx=5, fill=X, expand=True)
        self.test_ai_button = ttk.Button(
            test_ai_frame,
            text="Test AI",
            command=self.test_ai,
            style='TButton'
        )
        self.test_ai_button.pack(side=LEFT, padx=5)

        # JSON Data tab
        self.json_data_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.json_data_frame, text="JSON Data")

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
        self.ai_insights_frame.grid(row=1, column=0, sticky='ew')
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

    def update_recursive(self):
        self.base_tenancy_data.recursive = self.recursive_var.get()
        logging.info(f"Recursive setting updated to: {self.recursive_var.get()}")

    def test_ai(self):
        input_text = self.test_ai_var.get()
        if input_text:
            markdown_text = self.ai.test(input_text)
            html_content = markdown.markdown(markdown_text)
            self.ai_insights_text.set_html(html_content)
            logging.info("AI test completed and insights updated")
        else:
            logging.warning("No input provided for Test AI")

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
        auth = self.base_tenancy_data.load(mode, param)
        # Update AI client with authentication
        if auth:
            if mode == "Instance Principal":
                self.ai.signer = auth
                self.ai.config = None
            elif mode == "Profile":
                self.ai.config = auth
                self.ai.signer = None
            self.ai.initialize_client()
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
            self.console_frame.grid(row=2, column=0, sticky='ew')
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