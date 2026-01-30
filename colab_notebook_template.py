# Google Colab Notebook: Collective Learning Medical QA

"""
Collective Learning for Turkish Medical Question Answering
Project #9: Self-Improving Language Models
Optimized for Google Colab Free Tier

IMPORTANT: This notebook requires a Hugging Face token.
1. Create an account at https://huggingface.co
2. Generate a token at https://huggingface.co/settings/tokens
3. Add it to Colab secrets as 'HF_TOKEN'
"""

#%% [markdown]
# ## Step 1: Install Dependencies

#%%
!pip install -q transformers datasets peft accelerate bitsandbytes
!pip install -q huggingface_hub

#%% [markdown]
# ## Step 2: Setup Environment and Authentication

#%%
import os
from google.colab import userdata
import torch

# Set Hugging Face token from Colab secrets
# To add: Go to 🔑 (Secrets) in left sidebar → Add HF_TOKEN
try:
    os.environ["HF_TOKEN"] = userdata.get('HF_TOKEN')
    print("✓ HuggingFace token loaded from Colab secrets")
except:
    print("⚠️ Warning: HF_TOKEN not found in secrets")
    print("Please add your token to Colab secrets or set manually:")
    # Uncomment and add your token (NOT RECOMMENDED - use secrets instead):
    # os.environ["HF_TOKEN"] = "your_token_here"

# Check GPU
print(f"\nGPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

#%% [markdown]
# ## Step 3: Download Main Script

#%%
# Upload the collective_learning_medical_qa.py file
# Or copy the code directly here

from google.colab import files

print("Please upload the 'collective_learning_medical_qa.py' file")
uploaded = files.upload()

# Verify upload
if 'collective_learning_medical_qa.py' in uploaded:
    print("✓ Script uploaded successfully")
else:
    print("⚠️ Script not found. Please upload it.")

#%% [markdown]
# ## Step 4: Configure for Colab (Optional - Reduce Memory Usage)

#%%
# If you have memory issues, create a modified config
# This reduces batch size and LoRA rank to fit in free tier GPU

config_override = """
# Memory-optimized configuration for Google Colab
class ColabConfig(Config):
    BATCH_SIZE = 2  # Reduced from 4
    GRADIENT_ACCUMULATION_STEPS = 8  # Increased to maintain effective batch size
    LORA_R = 8  # Reduced from 16
    MAX_LENGTH = 256  # Reduced from 512
    TRAIN_SIZE_PER_ITER = 200  # Reduced from 500 for faster iteration
"""

# Uncomment and run this cell if you want to use reduced settings
# with open('collective_learning_medical_qa.py', 'a') as f:
#     f.write(config_override)

#%% [markdown]
# ## Step 5: Run the Experiment

#%%
# Full experiment (takes ~3-4 hours on Colab free tier)
!python collective_learning_medical_qa.py

#%% [markdown]
# ## Step 6: View Results

#%%
import json
import pandas as pd

# Load experimental results
with open('./collective_learning_outputs/experimental_results.json', 'r') as f:
    results = json.load(f)

# Display iteration statistics
print("=" * 60)
print("ITERATION STATISTICS")
print("=" * 60)
for iter_data in results['iterations']:
    print(f"Iteration {iter_data['iteration']}: Avg Score = {iter_data['avg_score']:.2f}/10")

print(f"\nFinal Test Set Score: {results['avg_test_score']:.2f}/10")

# Show sample test results
print("\n" + "=" * 60)
print("SAMPLE TEST RESULTS")
print("=" * 60)
for i, result in enumerate(results['test_results'][:3]):  # Show first 3
    print(f"\nQuestion {i+1}: {result['question']}")
    print(f"Answer: {result['answer'][:200]}...")
    print(f"Score: {result['evaluation'].get('genel_puan', 'N/A')}/10")

#%% [markdown]
# ## Step 7: Download Results

#%%
from google.colab import files

# Download all results
!zip -r results.zip ./collective_learning_outputs

print("Downloading results.zip...")
files.download('results.zip')

#%% [markdown]
# ## Alternative: Quick Test (5 Questions Only)

#%%
# For quick testing, modify the configuration

quick_test_code = """
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Quick test with 5 questions only
print("Running quick test with 5 questions...")

# Load model
model_name = "cosmocanyon/cosmos-T1-2B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# Load dataset
dataset = load_dataset("hoatac/Medikal-QA-Turkish")
questions = dataset['train']['question'][:5]

# Test answer generation
for i, question in enumerate(questions):
    prompt = f"Aşağıdaki tıbbi soruyu yanıtlayın.\\n\\nSoru: {question}\\n\\nYanıt:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100)
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\\nQ{i+1}: {question}")
    print(f"A{i+1}: {answer[len(prompt):].strip()[:150]}...")
    
print("\\n✓ Quick test complete!")
"""

# Uncomment to run quick test
# exec(quick_test_code)

#%% [markdown]
# ## Troubleshooting

# ### Out of Memory Error
# If you get CUDA out of memory:
# 1. Restart runtime (Runtime → Restart runtime)
# 2. Reduce batch size to 1
# 3. Reduce LORA_R to 4
# 4. Reduce MAX_LENGTH to 128
# 5. Use fewer training samples (200 per iteration)

# ### Model Loading Issues
# If model download fails:
# 1. Check your HF_TOKEN is valid
# 2. Ensure you've accepted model terms on HuggingFace
# 3. Try restarting runtime

# ### Dataset Loading Issues
# If dataset fails to load:
# 1. Check internet connection
# 2. Try: !pip install --upgrade datasets
# 3. Clear cache: !rm -rf ~/.cache/huggingface

#%% [markdown]
# ## Expected Timeline (Colab Free Tier)

# - Setup & Installation: ~5 minutes
# - Model Download: ~10 minutes
# - Dataset Loading: ~2 minutes
# - Iteration 0 (Generation + Evaluation + Improvement): ~30 minutes
# - Training Iteration 0→1: ~40 minutes
# - Iteration 1 (Generation + Evaluation + Improvement): ~30 minutes
# - Training Iteration 1→2: ~40 minutes
# - Iteration 2 (Generation + Evaluation + Improvement): ~30 minutes
# - Training Iteration 2→3: ~40 minutes
# - Final Evaluation: ~15 minutes
# 
# **Total: ~4 hours**

#%% [markdown]
# ## Notes

# - **Save Checkpoints**: Results are saved after each iteration in `./collective_learning_outputs/`
# - **Runtime Limits**: Colab free tier has ~12 hour limit; this experiment fits comfortably
# - **GPU Memory**: T4 (15GB) is sufficient; reduce batch size if needed
# - **Interruptions**: If interrupted, you can resume from last completed iteration
# - **Token Limit**: HuggingFace tokens are free; just need account

#%% [markdown]
# ## Citation

# If you use this notebook or the collective learning approach:

# ```
# Collective Learning for Turkish Medical Question Answering: 
# A Self-Improving Language Model Approach
# Project #9: Biomedical Signal & Image Processing
# 2026
# ```
