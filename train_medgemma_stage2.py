import os
import sys

# =====================================================================
# 1. RADICAL CORE CREDENTIAL SWEEP & LOCK
# =====================================================================
for key in list(os.environ.keys()):
    if "HF" in key or "TOKEN" in key:
        del os.environ[key]

my_secret_token = "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"
os.environ["HF_TOKEN"] = my_secret_token
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import huggingface_hub
from huggingface_hub import login

try:
    login(token=my_secret_token)
    huggingface_hub.constants.HF_TOKEN = my_secret_token
    print("Hugging Face Hub authenticated successfully with updated token layer.")
except Exception as e:
    print(f"Token binder active with notice: {e}")

import torch
import transformers
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig

model_id = "google/medgemma-27b-it" 
dataset_path = "6500_samples.json"  

if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset missing at {dataset_path}. Please run generate_data.py first!")

# =====================================================================
# 2. THE HOT-PATCH (Bypasses the library version collision)
# =====================================================================
_original_trainer_init = transformers.Trainer.__init__

def _patched_trainer_init(self, *args, **kwargs):
    if "tokenizer" in kwargs:
        kwargs["processing_class"] = kwargs.pop("tokenizer")
    _original_trainer_init(self, *args, **kwargs)

transformers.Trainer.__init__ = _patched_trainer_init
# =====================================================================

# 3. Setup High-Efficiency 4-Bit Quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16  
)

print("Loading processor and base model configuration onto the local RTX 6000...")

processor = AutoProcessor.from_pretrained(
    model_id,
    token=my_secret_token
)
processor.tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    token=my_secret_token
)

# 4. Setup Fresh LoRA Configuration 
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], 
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

print("Loading dataset...")
dataset = load_dataset("json", data_files=dataset_path, split="train")

# 5. Adjusted SFTConfig (Bare-metal compliant for any TRL version)
training_args = SFTConfig(
    output_dir="./medgemma-fresh-output",
    per_device_train_batch_size=4,        
    gradient_accumulation_steps=4,        
    optim="paged_adamw_8bit",             
    logging_steps=10,
    learning_rate=2e-4,                   
    bf16=True,                            
    max_grad_norm=0.3,
    num_train_epochs=2,                    
    warmup_ratio=0.03,
    save_strategy="no",                   
    gradient_checkpointing=True,          
    report_to="none"
)

# 6. Initialize Trainer 
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    processing_class=processor.tokenizer,  
    args=training_args
)

print("Engine stabilized. Commencing local fine-tuning loop...")
trainer.train()

# 7. Safe Manual Serialization & Export
print("Training complete! Forcing manual safe export of clean adapters...")
output_folder = "health_model"
os.makedirs(output_folder, exist_ok=True)

trainable_state_dict = {k: v for k, v in trainer.model.state_dict().items() if "lora_" in k}
torch.save(trainable_state_dict, os.path.join(output_folder, "adapter_model.bin"))

trainer.model.peft_config['default'].save_pretrained(output_folder)
processor.save_pretrained(output_folder)

print("All tasks completed successfully. Fresh local adapters fully saved.")
