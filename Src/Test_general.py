import os
import sys
import time
import csv
import base64
from pathlib import Path
from PIL import Image

# --- External Library Imports ---
try:
    # Hugging Face imports (for BLIP and VQA)
    import torch
    from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering
except ImportError:
    print("WARNING: 'torch' or 'transformers' not installed. BLIP/VQA models will be skipped.")
    torch = None

try:
    # Google Gemini API imports
    from google import genai
    from google.genai.errors import APIError
except ImportError:
    print("WARNING: 'google-genai' not installed. Gemini model will be skipped.")
    genai = None

try:
    # Ollama imports (for Gemma3 and Moondream)
    from ollama import chat
except ImportError:
    print("WARNING: 'ollama' Python client not installed. Ollama models will be skipped.")
    chat = None


# --- Configuration ---
WARMUP_REPETITIONS = 2
DEFAULT_REPETITIONS = 5
OUTPUT_CSV = "benchmark_results.csv"
# UPDATED PATH: The class CSV is now expected inside the image directory
CLASS_CSV = "Test_images/image_classes.csv"
DEVICE = "cuda" if torch and torch.cuda.is_available() else "cpu"

PROMPT_CAPTION = "I see a" 
PROMPT_DESCRIPTIVE = "Describe this image in detail, concisely, focusing on the main objects and colors." 

# VQA Questions adapted to run TWO questions per class and sum the time
# Each value is a list of (question, suffix) tuples
VQA_QUESTIONS = {
    "person": [
        {"q": "What is the hair color of the person in the image?", "suffix": "hair"},
        {"q": "What is the clothing color of the person in the image?", "suffix": "clothes"},
    ],
    "car": [
        {"q": "What is the color of the car in the image?", "suffix": "car"},
        {"q": "What is the type of the car that is shown in the image?", "suffix": ""},
    ],
    "truck": [
        {"q": "What is the color of the truck in the image?", "suffix": "truck"},
        {"q": "What is the type of the truck that is shown in the image?", "suffix": "truck"},
    ],
    "motorcycle": [
        {"q": "What is the color of the motorcycle in the image?", "suffix": "motorcycle"},
        {"q": "What is the type of the motorcycle that is shown in the image?", "suffix": ""},
    ],
    "bus": [
        {"q": "What is the color of the bus in the image?", "suffix": "bus"},
        {"q": "What is the type of the bus that is shown in the image?", "suffix": ""},
    ],
    # Default is handled separately as a single descriptive prompt
    "default": [{"q": PROMPT_DESCRIPTIVE, "suffix": ""}] 
}

# --- Utility Functions ---

def imagem_para_base64(caminho_imagem: str) -> str:
    """Reads an image and returns it in base64 format for Ollama."""
    with open(caminho_imagem, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

def log_result(writer, exec_num, image_name, model_name, time_ms, response_text):
    """Writes a single benchmark result row to the CSV."""
    # Sanitize response to ensure no newlines break the CSV row format
    sanitized_response = response_text.replace('\n', ' ').replace('\r', ' ').strip()
    writer.writerow([exec_num, image_name, model_name, f"{time_ms:.4f}", sanitized_response])

def load_image_classes(class_csv_path: Path) -> dict:
    """Loads a dictionary mapping image filenames to their object class from a CSV file."""
    class_map = {}
    if not class_csv_path.exists():
        print(f"WARNING: Class configuration file not found at '{class_csv_path}'. VQA will use default prompt for all images.")
        return class_map

    try:
        with open(class_csv_path, 'r', newline='') as f:
            reader = csv.reader(f)
            # Skip potential header row
            try:
                header = next(reader)
                if header[0].lower() != 'image_name': 
                    class_map[header[0]] = header[1]
            except StopIteration:
                return class_map
            for row in reader:
                if len(row) >= 2:
                    image_name = row[0].strip()
                    object_class = row[1].strip()
                    class_map[image_name] = object_class
    except Exception as e:
        print(f"ERROR: Could not read or parse '{class_csv_path}'. Error: {e}")
        
    return class_map

# --- Model Wrapper Classes ---
class BlipCaptioner:
    """Base class for BLIP models, handling deferred loading/unloading to manage VRAM."""
    def __init__(self, model_name, alias, model_class):
        self.alias = alias
        self.model_name = model_name
        self.model_class = model_class
        self.processor = None
        self.model = None
        self.status = "UNINITIALIZED"
        
        if not torch:
            self.status = "SKIPPED_DEP"
            return
        self.status = "READY_TO_LOAD"

    def load(self):
        """Loads model weights to the GPU (DEVICE)."""
        if self.status != "READY_TO_LOAD":
            print(f"  [STATUS] Skipping load for {self.alias} (Status: {self.status}).")
            return False
            
        print(f"  [VRAM] Loading {self.alias} ({self.model_name}) onto {DEVICE}...")
        try:
            self.processor = BlipProcessor.from_pretrained(self.model_name)
            self.model = self.model_class.from_pretrained(self.model_name).to(DEVICE)
            self.status = "LOADED"
            print(f"  [VRAM] {self.alias} loaded successfully.")
            return True
        except Exception as e:
            self.status = "FAILED_LOAD"
            print(f"  [VRAM] Failed to load {self.alias}: {e}")
            return False

    def unload(self):
        """Removes model weights from the GPU and clears cache."""
        if self.status == "LOADED":
            print(f"  [VRAM] Unloading {self.alias} and clearing cache...")
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            self.status = "READY_TO_LOAD"

    def run_inference(self, image_path: Path, object_class: str = 'default'):
        """Performs a single inference and returns (time_s, response_text) or error code."""
        if self.status != "LOADED":
            return -1 # Error code

        try:
            raw_image = Image.open(image_path).convert('RGB')
            question = PROMPT_CAPTION
            
            start_time = time.perf_counter()
            inputs = self.processor(raw_image, question, return_tensors="pt").to(DEVICE)
            outputs = self.model.generate(**inputs, max_new_tokens=50) 
            end_time = time.perf_counter()

            response_text = self.processor.decode(outputs[0], skip_special_tokens=True)
            return (end_time - start_time), response_text # Success tuple
            
        except Exception as e:
            print(f"Error during {self.alias} inference on {image_path}: {e}")
            return -2 # Error code

class BlipBaseCaptioner(BlipCaptioner):
    def __init__(self):
        super().__init__("Salesforce/blip-image-captioning-base", "BLIP-Base", BlipForConditionalGeneration)

class BlipLargeCaptioner(BlipCaptioner):
    def __init__(self):
        super().__init__("Salesforce/blip-image-captioning-large", "BLIP-Large", BlipForConditionalGeneration)

class BlipVQACaptioner(BlipCaptioner):
    """Wrapper for BLIP VQA model, using class-specific questions if available."""
    def __init__(self):
        super().__init__("Salesforce/blip-vqa-base", "BLIP-VQA", BlipForQuestionAnswering)

    def run_inference(self, image_path: Path, object_class: str = 'default'):
        """Performs VQA inference for all questions associated with the class, summing time and responses."""
        if self.status != "LOADED":
            return -1

        try:
            raw_image = Image.open(image_path).convert('RGB')
            
            # Select question list based on the class, defaulting to a single descriptive prompt
            question_list = VQA_QUESTIONS.get(object_class.lower(), VQA_QUESTIONS["default"])

            total_time = 0.0
            full_response = []
            
            # Loop through all questions for the identified class
            for item in question_list:
                question = item["q"]
                suffix = item["suffix"]

                start_time = time.perf_counter()
                inputs = self.processor(raw_image, question, return_tensors="pt").to(DEVICE)
                outputs = self.model.generate(**inputs, max_new_tokens=50)
                elapsed_time = time.perf_counter() - start_time
                total_time += elapsed_time

                answer = self.processor.decode(outputs[0], skip_special_tokens=True).strip().rstrip(".")

                # Apply suffix logic from original vqa2.py for single-word answers
                if len(answer.split()) == 1 and suffix:
                    answer = f"{answer} {suffix}"
                
                # Format response for CSV logging: "Q1: Answer1 | Q2: Answer2"
                full_response.append(f"{question}: {answer}")

            response_text = " | ".join(full_response)
            return total_time, response_text # Success tuple (summed time)
            
        except Exception as e:
            print(f"Error during {self.alias} inference on {image_path}: {e}")
            return -2

class ApiCpuModel:
    """Base class for models not requiring sequential VRAM management (Gemini/Ollama)."""
    def __init__(self, alias):
        self.alias = alias
        self.status = "UNINITIALIZED"
    
    def load(self):
        """Initializes the model client/functionality."""
        if self.status == "UNINITIALIZED":
             self._initialize()
        return self.status == "LOADED"

    def unload(self):
        """No-op for API/CPU models."""
        pass 

    def _initialize(self):
        """Internal initialization logic defined by subclasses."""
        self.status = "LOADED" # Assume success by default

    def run_inference(self, image_path: Path, object_class: str = 'default'):
        """Overridden by subclasses."""
        raise NotImplementedError("Subclasses must implement run_inference.")

class GeminiFlashCaptioner(ApiCpuModel):
    def __init__(self):
        super().__init__("Gemini-2.0-Flash")
        self.model_name = "gemini-2.0-flash"
        self.client = None

    def _initialize(self):
        if not genai:
            self.status = "SKIPPED_DEP"
            return

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("WARNING: GEMINI_API_KEY not set. Skipping Gemini model.")
            self.status = "SKIPPED_KEY"
            return

        try:
            self.client = genai.Client()
            self.status = "LOADED"
            print(f"  [API] {self.alias} client initialized.")
        except Exception as e:
            self.client = None
            self.status = "FAILED_LOAD"
            print(f"Failed to initialize Gemini client: {e}")

    def run_inference(self, image_path: Path, object_class: str = 'default'): 
        """Performs a single Gemini API call and returns (time_s, response_text) or error code."""
        if self.status != "LOADED":
            return -1

        try:
            img = Image.open(image_path)
            contents = [img, PROMPT_DESCRIPTIVE]
            
            start_time = time.perf_counter()
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents
            )
            end_time = time.perf_counter()
            
            response_text = response.text
            return (end_time - start_time), response_text # Success tuple
            
        except APIError as e:
            print(f"Gemini API Error on {image_path}: {e}")
            return -3
        except Exception as e:
            print(f"Error during {self.alias} inference on {image_path}: {e}")
            return -2

class OllamaCaptioner(ApiCpuModel):
    def __init__(self, model_name, alias):
        super().__init__(alias)
        self.model_name = model_name
        self.chat_func = chat if chat else None

    def _initialize(self):
        if not self.chat_func:
            self.status = "SKIPPED_DEP"
            return
        
        self.status = "LOADED"
        print(f"  [CPU] Ollama client initialized for {self.alias}.")

    def run_inference(self, image_path: Path, object_class: str = 'default'): 
        """Performs a single Ollama chat call and returns (time_s, response_text) or error code."""
        if self.status != "LOADED":
            return -1

        try:
            img_base64 = imagem_para_base64(str(image_path))
            
            start_time = time.perf_counter()
            response_data = self.chat_func(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": PROMPT_DESCRIPTIVE, "images": [img_base64]}
                ]
            )
            end_time = time.perf_counter()

            response_text = response_data["message"]["content"]
            return (end_time - start_time), response_text # Success tuple
            
        except Exception as e:
            print(f"Error during {self.alias} inference on {image_path}. Check if Ollama server is running and model '{self.model_name}' is downloaded. Error: {e}")
            return -2

class OllamaGemmaCaptioner(OllamaCaptioner):
    def __init__(self):
        super().__init__("gemma3:4b", "Ollama-Gemma3-4B")

class OllamaMoondreamCaptioner(OllamaCaptioner):
    def __init__(self):
        super().__init__("moondream", "Ollama-Moondream")

# --- Main Benchmarking Logic ---

def run_test_sequence(model_instance, image_paths, image_to_class_map, repetitions, writer, total_execution_offset):
    """Handles warmup and timed repetitions for a single, already-loaded model."""
    
    total_execution_count = total_execution_offset
    
    # 1. Warmup Phase (across all images)
    print(f"    --- Starting Warmup ({WARMUP_REPETITIONS} runs per image) ---")
    
    for image_path in image_paths:
        image_name = image_path.name
        object_class = image_to_class_map.get(image_name, 'default').lower()
        class_to_pass = object_class if model_instance.alias == "BLIP-VQA" else 'default'
        
        # Run warmups
        for w in range(1, WARMUP_REPETITIONS + 1):
            result = model_instance.run_inference(image_path, class_to_pass)
            
            # Extract time: result is either an error code (int) or (time, response) (tuple)
            time_s = result[0] if isinstance(result, tuple) else result
            
            if time_s < 0:
                print(f"    [FAIL] Warmup failed on {image_name} with error code {time_s}. Skipping timed runs for this model.")
                return 0 
            print(f"    (Warmup {w}/{WARMUP_REPETITIONS} time on {image_name}: {time_s:.4f}s)")
    
    # 2. Timed Repetition Phase (across all images)
    print(f"    --- Starting Timed Repetitions ({repetitions} runs per image) ---")

    for image_path in image_paths:
        image_name = image_path.name
        object_class = image_to_class_map.get(image_name, 'default').lower()
        class_to_pass = object_class if model_instance.alias == "BLIP-VQA" else 'default'

        for r in range(1, repetitions + 1):
            result = model_instance.run_inference(image_path, class_to_pass)
            
            if isinstance(result, tuple):
                time_s, response_text = result
                # Log result
                total_execution_count += 1
                log_result(writer, total_execution_count, image_name, model_instance.alias, time_s, response_text)
                print(f"    [LOG] Exec {total_execution_count:04d} on {image_name}: {time_s:.4f}s | Response logged.")
            else:
                time_s = result # Time_s is the error code
                print(f"    [FAIL] Timed run {r}/{repetitions} failed on {image_name} with error code {time_s}.")
                
    return total_execution_count - total_execution_offset

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_benchmark.py <path_to_image_directory> [repetitions]")
        print("Example: python run_benchmark.py ./test_images/ 10")
        sys.exit(1)

    image_dir = Path(sys.argv[1])
    if not image_dir.is_dir():
        print(f"Error: Directory not found at '{image_dir}'")
        sys.exit(1)

    try:
        repetitions = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_REPETITIONS
    except ValueError:
        print("Error: Repetitions must be an integer.")
        sys.exit(1)

    # 1. Load image class mappings
    # We construct the full path for the class CSV relative to the current directory,
    # then use the path of the image directory argument if it's different.
    # NOTE: Path() handles cross-platform path construction
    class_csv_path = Path(image_dir) / Path(CLASS_CSV).name # Ensure we are reading from Test_images/image_classes.csv
    image_to_class_map = load_image_classes(class_csv_path)

    # 2. Define Model Groups (Instances are created but not loaded yet)
    all_models = [
        #BlipBaseCaptioner(),
        BlipLargeCaptioner(),
        BlipVQACaptioner(),
        #GeminiFlashCaptioner(),
        #OllamaGemmaCaptioner(),
        #OllamaMoondreamCaptioner(),
    ]
    
    gpu_models = [m for m in all_models if isinstance(m, BlipCaptioner)]
    api_cpu_models = [m for m in all_models if isinstance(m, ApiCpuModel)]
    
    # 3. Find images
    image_paths = sorted([p for p in image_dir.glob("*") if p.suffix.lower() in ('.jpg', '.jpeg', '.png')])
    
    if not image_paths:
        print(f"No JPG/PNG images found in '{image_dir}'")
        sys.exit(1)

    print(f"\n--- Starting Benchmark ---")
    print(f"Target Images: {len(image_paths)}")
    print(f"Warmup Runs: {WARMUP_REPETITIONS}")
    print(f"Timed Repetitions: {repetitions}")
    print(f"GPU Device: {DEVICE}")

    # 4. Setup CSV writer
    with open(OUTPUT_CSV, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # UPDATED CSV HEADER
        writer.writerow(["Execution_Number", "Image_Name", "Model", "Time_to_Response_s", "Model_Response"])

        total_logs_written = 0
        
        # --- PHASE A: Run API/CPU Models (Initialized once, run all tests) ---
        print("\n--- PHASE A: Initializing API/CPU Models ---")
        for model in api_cpu_models:
            model.load()

        print("\n--- PHASE A: Running API/CPU Tests ---")
        for model in api_cpu_models:
            if model.status == "LOADED":
                print(f"\n[MODEL] Starting tests for {model.alias}...")
                new_logs = run_test_sequence(model, image_paths, image_to_class_map, repetitions, writer, total_logs_written)
                total_logs_written += new_logs
            else:
                print(f"\n[MODEL] Skipping {model.alias} (Status: {model.status}).")


        # --- PHASE B: Run GPU Models (Loaded sequentially) ---
        print("\n--- PHASE B: Running GPU Tests (Sequential Loading) ---")
        for model in gpu_models:
            if model.status == "SKIPPED_DEP":
                print(f"\n[MODEL] Skipping {model.alias} due to missing dependencies.")
                continue

            print(f"\n[MODEL] Starting sequential tests for {model.alias}...")
            
            # Load model to VRAM
            if model.load():
                # Run full test sequence
                new_logs = run_test_sequence(model, image_paths, image_to_class_map, repetitions, writer, total_logs_written)
                total_logs_written += new_logs
                # Unload model from VRAM
                model.unload()
            else:
                print(f"[MODEL] Skipping {model.alias} due to loading failure.")

    print(f"\n--- Benchmark Finished ---")
    print(f"Total executions logged: {total_logs_written}")
    print(f"Results saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()