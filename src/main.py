# AI Endpoint working and saved options
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

# Region to Generative AI endpoint mapping
REGION_TO_AI_ENDPOINT = {
    'us-chicago-1': 'https://inference.generativeai.us-chicago-1.oci.oraclecloud.com',
    'us-ashburn-1': 'https://inference.generativeai.us-ashburn-1.oci.oraclecloud.com',
    'eu-frankfurt-1': 'https://inference.generativeai.eu-frankfurt-1.oci.oraclecloud.com',
    'uk-london-1': 'https://inference.generativeai.uk-london-1.oci.oraclecloud.com',
    # Add more regions as needed
}

# AI module
class AI:
    def __init__(self, parent, model, endpoint, compartment_id, config=None, signer=None):
        self.parent = parent
        self.model = model
        self.endpoint = endpoint
        self.compartment_id = compartment_id
        self.config = config
        self.signer = signer
        self.client = None
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
        logging.info(f"AI module initialized with model: {self.model}, endpoint: {self.endpoint}, compartment_id: {self.compartment_id}")

    def initialize_client(self):
        try:
            if not self.compartment_id:
                raise ValueError("Compartment OCID is required")
            if self.signer:
                self.client = oci.generative_ai_inference.GenerativeAiInferenceClient(
                    config={}, signer=self.signer, service_endpoint=self.endpoint
                )
            elif self.config:
                self.client = oci.generative_ai_inference.GenerativeAiInferenceClient(
                    config=self.config, service_endpoint=self.endpoint
                )
            else:
                raise ValueError("No valid authentication provided")
            logging.info("AI client initialized successfully")
            return True
        except Exception as e:
            logging.error(f"Failed to initialize AI client: {e}")
            self.client = None
            return False

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
        self.parent.save_options()

    def update_endpoint(self, endpoint):
        self.endpoint = endpoint
        if self.initialize_client():
            self.parent.save_options()
        logging.info(f"AI endpoint updated to: {self.endpoint}")

    def update_compartment_id(self, compartment_id):
        self.compartment_id = compartment_id
        if self.initialize_client():
            self.parent.save_options()
        logging.info(f"AI compartment ID updated to: {self.compartment_id}")

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
        if not self.initialize_client():
            return "Error: Failed to initialize AI client"
        result = self.query_genai(
            prompt=input_text,
            cache_type="test_ai",
            cache_query=input_text,
            cache=self.cache
        )
        if not result.startswith("Error:"):
            self.parent.save_options()
        return result

# BaseTenancyData module
class BaseTenancyData:
    def __init__(self, parent, recursive):
        self.parent = parent
        self.recursive = recursive
        self.data = {}
        self.region = None
        logging.info(f"BaseTenancyData module initialized with recursive: {self.recursive}")

    def load(self, mode, param=None):
        self.data = {}
        self.region = None
        if mode == "Instance Principal":
            try:
                signer = oci.auth.signers.InstancePrincipalsSecurityTokenSigner()
                client = oci.identity.IdentityClient(config={}, signer=signer)
                tenancy = client.get_tenancy(tenancy_id=signer.tenancy_id).data
                self.data = {"tenancy": {"name": tenancy.name, "id": tenancy.id}}
                self.region = signer.region
                logging.info(f"Loaded tenancy data using Instance Principal, region: {self.region}")
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
                self.region = config.get('region', 'us-ashburn-1')
                logging.info(f"Loaded tenancy data using Profile: {param}, region: {self.region}")
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
                self.region = 'us-ashburn-1'  # Default region for cache mode
                logging.info(f"Loaded tenancy data from Cache: {param}, region: {self.region}")
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
        self.options_file = 'cache/options.json'

        # Variables
        self.ai_model_var = tk.StringVar(value="default_model")
        self.ai_endpoint_var = tk.StringVar(value=REGION_TO_AI_ENDPOINT.get('us-ashburn-1', 'https://api.example.com'))
        self.compartment_ocid_var = tk.StringVar(value="default-compartment-id")
        self.ai_enabled_var = tk.BooleanVar(value=True)
        self.load_from_var = tk.StringVar(value="Instance Principal")
        self.secondary_var = tk.StringVar()
        self.show_console_var = tk.BooleanVar(value=False)
        self.view_mode_var = tk.DoubleVar(value=0.0)  # Slider: 0=Tree, 1=Table
        self.recursive_var = tk.BooleanVar(value=False)  # Recursive checkbox
        self.test_ai_var = tk.StringVar(value="")  # Test AI input
        self.log_level_var = tk.StringVar(value='INFO')

        # Load saved options
        self.load_options()

        # Initialize modules
        self.base_tenancy_data = BaseTenancyData(self, self.recursive_var.get())
        self.ai = AI(self, self.ai_model_var.get(), self.ai_endpoint_var.get(), self.compartment_ocid_var.get())

        # Configure grid
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
        self.grid_columnconfigure(0, weight=1)

        # Styles for visibility
        style = ttk.Style()
        style.configure('TLabel', font=('Helvetica', 10), foreground='black', background='white')
        style.configure('TButton', font=('Helvetica', 10), foreground='black', background='lightgray', bootstyle=INFO)
        style.configure('TCheckbutton', font=('Helvetica', 10), foreground='black', background='white')
        style.configure('TCombobox', font=('Helvetica', 10), foreground='black', background='white')
        style.configure('TEntry', font=('Helvetica', 10), foreground='black', background='white')
        style.configure('TLabelFrame.Label', font=('Helvetica', 10, 'bold'), foreground='black', background='white')
        style.configure('TNotebook', bootstyle=INFO)

        # Main notebook
        self.notebook = ttk.Notebook(self, style='TNotebook')
        self.notebook.grid(row=0, column=0, sticky='nsew')

        # START tab
        self.start_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.start_frame, text="START")

        # General group
        general_frame = ttk.LabelFrame(self.start_frame, text="General", padding=10)
        general_frame.pack(fill=X, padx=10, pady=5)
        ttk.Label(general_frame, text="Configure general application settings.", font=('Helvetica', 10)).pack(anchor='w', pady=2)
        console_log_frame = ttk.Frame(general_frame)
        console_log_frame.pack(fill=X, pady=2)
        self.console_toggle = ttk.Checkbutton(
            console_log_frame,
            text="Show Console Log",
            variable=self.show_console_var,
            command=self.toggle_console,
            style='TCheckbutton'
        )
        self.console_toggle.pack(side=LEFT, padx=5)
        log_level_frame = ttk.Frame(console_log_frame)
        log_level_frame.pack(side=LEFT, padx=5)
        ttk.Label(log_level_frame, text="Log Level:", style='TLabel').pack(side=LEFT)
        self.log_level_combo = ttk.Combobox(
            log_level_frame,
            textvariable=self.log_level_var,
            values=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
            state='readonly',
            style='TCombobox',
            width=10
        )
        self.log_level_combo.pack(side=LEFT, padx=5)
        self.log_level_combo.bind('<<ComboboxSelected>>', self.change_log_level)

        # Tenancy group
        tenancy_frame = ttk.LabelFrame(self.start_frame, text="Tenancy", padding=10)
        tenancy_frame.pack(fill=X, padx=10, pady=5)
        ttk.Label(tenancy_frame, text="Select the tenancy data source and options.", font=('Helvetica', 10)).pack(anchor='w', pady=2)
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
        recursive_load_frame = ttk.Frame(tenancy_frame)
        recursive_load_frame.pack(fill=X, pady=2)
        recursive_check = ttk.Checkbutton(
            recursive_load_frame,
            text="Recursive",
            variable=self.recursive_var,
            command=self.update_recursive,
            style='TCheckbutton'
        )
        recursive_check.pack(side=LEFT, padx=5)
        self.load_button = ttk.Button(
            recursive_load_frame,
            text="Load",
            command=self.load_data,
            style='TButton',
            bootstyle=INFO
        )
        self.load_button.pack(side=LEFT, padx=5)

        # AI Config group
        ai_config_frame = ttk.LabelFrame(self.start_frame, text="AI Config", padding=10)
        ai_config_frame.pack(fill=X, padx=10, pady=5)
        ttk.Label(ai_config_frame, text="Configure AI model, endpoint, and compartment for insights.", font=('Helvetica', 10)).pack(anchor='w', pady=2)
        ai_enabled_frame = ttk.Frame(ai_config_frame)
        ai_enabled_frame.pack(fill=X, pady=2)
        self.ai_enabled_check = ttk.Checkbutton(
            ai_enabled_frame,
            text="AI Enabled",
            variable=self.ai_enabled_var,
            command=self.toggle_ai_enabled,
            style='TCheckbutton'
        )
        self.ai_enabled_check.pack(anchor='w', padx=5)
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
        compartment_frame = ttk.Frame(ai_config_frame)
        compartment_frame.pack(fill=X, pady=2)
        ttk.Label(compartment_frame, text="Compartment OCID:", style='TLabel').pack(side=LEFT, padx=5)
        self.compartment_ocid_entry = ttk.Entry(compartment_frame, textvariable=self.compartment_ocid_var, style='TEntry')
        self.compartment_ocid_entry.pack(side=LEFT, padx=5, fill=X, expand=True)
        self.compartment_ocid_entry.bind("<KeyRelease>", self.update_compartment_ocid)
        test_ai_frame = ttk.Frame(ai_config_frame)
        test_ai_frame.pack(fill=X, pady=2)
        ttk.Label(test_ai_frame, text="Test AI Input:", style='TLabel').pack(side=LEFT, padx=5)
        self.test_ai_entry = ttk.Entry(test_ai_frame, textvariable=self.test_ai_var, style='TEntry')
        self.test_ai_entry.pack(side=LEFT, padx=5, fill=X, expand=True)
        self.test_ai_button = ttk.Button(
            test_ai_frame,
            text="Test and Save",
            command=self.test_ai,
            style='TButton',
            bootstyle=INFO
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
        markdown_text = "# AI Insights\n**Model**: {}\n**Endpoint**: {}\n**Compartment OCID**: {}\nSample *italic* text.".format(
            self.ai_model_var.get(), self.ai_endpoint_var.get(), self.compartment_ocid_var.get()
        )
        html_content = markdown.markdown(markdown_text)
        self.ai_insights_text = HTMLScrolledText(self.ai_insights_frame, height=11)  # Increased by 10%
        self.ai_insights_text.pack(fill=BOTH, expand=True)
        self.ai_insights_text.set_html(html_content)

        # Console Log Frame
        self.console_frame = ttk.Frame(self, padding=5)
        self.clear_button = ttk.Button(
            self.console_frame,
            text="Clear",
            command=self.clear_console,
            style='TButton',
            bootstyle=INFO
        )
        self.clear_button.pack(side=LEFT, padx=5)
        self.console_text = ScrolledText(self.console_frame, height=10, autohide=True, font=('Helvetica', 10))
        self.console_text.pack(side=LEFT, fill=BOTH, expand=True)
        self.console_text.text.configure(background='white', foreground='black')

        self.text_handler = TextHandler(self.console_text)
        logging.getLogger().addHandler(self.text_handler)

        # Apply saved console visibility and AI enabled state
        self.toggle_console()
        self.toggle_ai_enabled()

        # Initial setup (populate dropdowns but don't load data)
        self.update_load_from(no_load=True)

        # Start background thread after UI initialization
        self.after(100, self.start_background_thread)

    def load_options(self):
        try:
            if os.path.exists(self.options_file):
                with open(self.options_file, 'r') as f:
                    options = json.load(f)
                self.ai_model_var.set(options.get('ai_model', 'default_model'))
                self.ai_endpoint_var.set(options.get('ai_endpoint', REGION_TO_AI_ENDPOINT.get('us-ashburn-1', 'https://api.example.com')))
                self.compartment_ocid_var.set(options.get('compartment_ocid', 'default-compartment-id'))
                self.ai_enabled_var.set(options.get('ai_enabled', True))
                self.load_from_var.set(options.get('load_from', 'Instance Principal'))
                self.secondary_var.set(options.get('secondary', ''))
                self.recursive_var.set(options.get('recursive', False))
                self.log_level_var.set(options.get('log_level', 'INFO'))
                self.show_console_var.set(options.get('show_console', False))
                logging.getLogger().setLevel(getattr(logging, self.log_level_var.get()))
        except Exception as e:
            logging.error(f"Failed to load options: {e}")

    def save_options(self):
        try:
            cache_dir = 'cache'
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            options = {
                'ai_model': self.ai_model_var.get(),
                'ai_endpoint': self.ai_endpoint_var.get(),
                'compartment_ocid': self.compartment_ocid_var.get(),
                'ai_enabled': self.ai_enabled_var.get(),
                'load_from': self.load_from_var.get(),
                'secondary': self.secondary_var.get(),
                'recursive': self.recursive_var.get(),
                'log_level': self.log_level_var.get(),
                'show_console': self.show_console_var.get()
            }
            with open(self.options_file, 'w') as f:
                json.dump(options, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save options: {e}")

    def start_background_thread(self):
        self.processing_thread = threading.Thread(target=self.process_queue, daemon=True)
        self.processing_thread.start()

    def update_ai_model(self, event=None):
        if self.ai_enabled_var.get():
            self.ai.update_model(self.ai_model_var.get())
            self.update_ai_insights()

    def update_ai_endpoint(self, event=None):
        if self.ai_enabled_var.get():
            self.ai.update_endpoint(self.ai_endpoint_var.get())
            self.update_ai_insights()

    def update_compartment_ocid(self, event=None):
        if self.ai_enabled_var.get():
            self.ai.update_compartment_id(self.compartment_ocid_var.get())
            self.update_ai_insights()

    def update_recursive(self):
        self.base_tenancy_data.recursive = self.recursive_var.get()
        logging.info(f"Recursive setting updated to: {self.recursive_var.get()}")
        self.save_options()

    def toggle_ai_enabled(self):
        if self.ai_enabled_var.get():
            self.ai_insights_frame.grid(row=1, column=0, sticky='ew')
            self.ai_model_entry['state'] = 'normal'
            self.ai_endpoint_entry['state'] = 'normal'
            self.compartment_ocid_entry['state'] = 'normal'
            self.test_ai_entry['state'] = 'normal'
            self.test_ai_button['state'] = 'normal'
        else:
            self.ai_insights_frame.grid_remove()
            self.ai_model_entry['state'] = 'disabled'
            self.ai_endpoint_entry['state'] = 'disabled'
            self.compartment_ocid_entry['state'] = 'disabled'
            self.test_ai_entry['state'] = 'disabled'
            self.test_ai_button['state'] = 'disabled'
        self.save_options()

    def test_ai(self):
        if not self.ai_enabled_var.get():
            logging.warning("AI is disabled; cannot test")
            return
        input_text = self.test_ai_var.get()
        if input_text:
            markdown_text = self.ai.test(input_text)
            html_content = markdown.markdown(markdown_text)
            self.ai_insights_text.set_html(html_content)
            logging.info("AI test completed and insights updated")
        else:
            logging.warning("No input provided for Test AI")

    def update_ai_insights(self):
        markdown_text = "# AI Insights\n**Model**: {}\n**Endpoint**: {}\n**Compartment OCID**: {}\nSample *italic* text.".format(
            self.ai_model_var.get(), self.ai_endpoint_var.get(), self.compartment_ocid_var.get()
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
                values = [f for f in os.listdir(cache_dir) if f.endswith('.json') and f != 'options.json']
        self.secondary_combo['values'] = values
        if values:
            saved_secondary = self.secondary_var.get()
            if saved_secondary in values:
                self.secondary_var.set(saved_secondary)
            else:
                self.secondary_var.set(values[0])
            self.secondary_combo['state'] = 'readonly'
        if not no_load:
            self.load_data()
        self.save_options()

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
        # Set AI endpoint based on region if not already set
        if self.base_tenancy_data.region:
            default_endpoint = REGION_TO_AI_ENDPOINT.get(self.base_tenancy_data.region, self.ai_endpoint_var.get())
            if self.ai_endpoint_var.get() == REGION_TO_AI_ENDPOINT.get('us-ashburn-1', 'https://api.example.com'):
                self.ai_endpoint_var.set(default_endpoint)
                self.ai.update_endpoint(default_endpoint)
        self.update_view()
        self.save_options()

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
        self.save_options()

    def change_log_level(self, event=None):
        level = self.log_level_var.get()
        logging.getLogger().setLevel(getattr(logging, level))
        logging.info(f"Log level changed to {level}")
        self.save_options()

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
        self.save_options()
        super().destroy()

if __name__ == "__main__":
    app = App()
    app.mainloop()