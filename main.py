import logging

logging.basicConfig(level=logging.INFO)  # or DEBUG


from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# # Get the token
hf_token = os.getenv("HF_TOKEN")

# Use the token (e.g., for login)
from huggingface_hub import login
login(token=hf_token)



TASK_LIST = ["Content Consumption", "Lifestyle Optimization", "Career Development"]
TASK = TASK_LIST[1]

# LLM = "Mistral-7B-Instruct-v0.2"      # Mistral/
# LLM = "Qwen2.5-3B-Instruct"             # Qwen/
# LLM = "Phi-4-mini-instruct"             # microsoft/
# LLM = "Phi-3-mini-128k-instruct"
LLM = "Llama-3.1-8B-Instruct"

USER_ID = "user_1"

USER_PERSONA = """
    - Likes green tea (Topic: Food)
    - Prefers matcha (Topic: Food)
    - Enjoys kombucha (Topic: Food)
    """
    
    
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# Define model ID and local paths
model_id = f"mistralai/{LLM}"
# model_id = f"meta-llama/{LLM}"
# model_id = f"microsoft/{LLM}"


MODEL_PATH = f"LLMs/{LLM}"
TOKENIZER_PATH = f"Tokenizers/{LLM}"

# Function to check if directory is already populated
def is_downloaded(directory):
    return os.path.exists(directory) and any(os.scandir(directory))

# Create folders if necessary
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(TOKENIZER_PATH, exist_ok=True)

# Download tokenizer if not already present
if not is_downloaded(TOKENIZER_PATH):
    print("⬇️ Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.save_pretrained(TOKENIZER_PATH)
    print(f"✅ Tokenizer saved to: {TOKENIZER_PATH}")
else:
    print(f"✅ Tokenizer already exists at: {TOKENIZER_PATH}")

# Download model if not already present
if not is_downloaded(MODEL_PATH):
    print("⬇️ Downloading model...")
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    model.save_pretrained(MODEL_PATH)
    print(f"✅ Model saved to: {MODEL_PATH}")
else:
    print(f"✅ Model already exists at: {MODEL_PATH}")
    
    
from agents.open_src_agent import PersoAgent

agent = PersoAgent(
    model_path = MODEL_PATH,
    tokenizer_path = TOKENIZER_PATH,
    user_id = USER_ID,
    task = TASK,
    prev_personas = USER_PERSONA
)


# Type 1: preference
response = agent.handle_task("I also enjoy soup for dinner.")

print(response)