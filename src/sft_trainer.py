# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, HfArgumentParser, TrainingArguments

from trl import SFTTrainer


tqdm.pandas()


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with SFTTrainer
    """

    model_name: Optional[str] = field(default="", metadata={"help": "the model name"})
    dataset_name: Optional[str] = field(
        default="", metadata={"help": "the dataset name"}
    )
    config_json: Optional[str] = field(default=None, metadata={"help": "the config json for PEFT"})
    dataset_text_field: Optional[str] = field(default="text", metadata={"help": "the text field of the dataset"})
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=3e-4, metadata={"help": "the learning rate"})
    batch_size: Optional[int] = field(default=64, metadata={"help": "the batch size"})
    seq_length: Optional[int] = field(default=128, metadata={"help": "Input sequence length"})
    gradient_accumulation_steps: Optional[int] = field(
        default=16, metadata={"help": "the number of gradient accumulation steps"}
    )
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 4 bits precision"})
    use_peft: Optional[bool] = field(default=False, metadata={"help": "Wether to use PEFT or not to train adapters"})
    trust_remote_code: Optional[bool] = field(default=True, metadata={"help": "Enable `trust_remote_code`"})
    output_dir: Optional[str] = field(default="output", metadata={"help": "the output directory"})
    peft_lora_r: Optional[int] = field(default=8, metadata={"help": "the r parameter of the LoRA adapters"})
    peft_lora_alpha: Optional[int] = field(default=32, metadata={"help": "the alpha parameter of the LoRA adapters"})
    logging_steps: Optional[int] = field(default=1, metadata={"help": "the number of logging steps"})
    use_auth_token: Optional[bool] = field(default=True, metadata={"help": "Use HF auth token to access the model"})
    local_files_only: Optional[bool] = field(default=False, metadata={"help": "Load only local files"})
    num_train_epochs: Optional[int] = field(default=10, metadata={"help": "the number of training epochs"})
    max_steps: Optional[int] = field(default=-1, metadata={"help": "the number of training steps"})
    save_steps: Optional[int] = field(
        default=100, metadata={"help": "Number of updates steps before two checkpoint saves"}
    )
    save_total_limit: Optional[int] = field(default=5, metadata={"help": "Limits total number of checkpoints."})
    push_to_hub: Optional[bool] = field(default=False, metadata={"help": "Push the model to HF Hub"})
    hub_model_id: Optional[str] = field(default=None, metadata={"help": "The name of the model on HF Hub"})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

print("loading model...")
# Step 1: Load the model
if script_args.load_in_8bit and script_args.load_in_4bit:
    raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
elif script_args.load_in_8bit or script_args.load_in_4bit:
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=script_args.load_in_8bit, load_in_4bit=script_args.load_in_4bit
    )
    # Copy the model to each device
    device_map = {"": Accelerator().local_process_index}
    torch_dtype = torch.bfloat16
else:
    quantization_config = None
    torch_dtype = None
    device_map = 'auto'

model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name,
    quantization_config=quantization_config,
    device_map=device_map,
    trust_remote_code=script_args.trust_remote_code,
    torch_dtype=torch_dtype,
    use_auth_token=script_args.use_auth_token,
    local_files_only=script_args.local_files_only,
)
model.config.use_cache = False

if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()
else:
    def make_inputs_require_grad(module, input, output):
         output.requires_grad_(True)

    model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

print("loading dataset...")
# Step 2: Load the dataset
try:
    # import pdb; pdb.set_trace()
    dataset = load_dataset(script_args.dataset_name, split="train")
except:
    dataset = load_dataset('csv', data_files=script_args.dataset_name)['train']
print("loading training args...")
# Step 3: Define the training arguments
training_args = TrainingArguments(
    output_dir=script_args.output_dir,
    per_device_train_batch_size=script_args.batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    learning_rate=script_args.learning_rate,
    logging_steps=script_args.logging_steps,
    num_train_epochs=script_args.num_train_epochs,
    max_steps=script_args.max_steps,
    report_to=script_args.log_with,
    save_steps=script_args.save_steps,
    save_total_limit=script_args.save_total_limit,
    push_to_hub=script_args.push_to_hub,
    hub_model_id=script_args.hub_model_id,
    gradient_checkpointing=True
)
print("loading LORA config...")
# Step 4: Define the LoraConfig
if script_args.use_peft and script_args.config_json is not None:
    config_json = json.load(open(script_args.config_json))
    peft_config = LoraConfig(**config_json)

elif script_args.use_peft and script_args.config_json is None:
    print("Using default PEFT config but no provided config_json. Reverting to provided values.")
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        inference_mode=False,
        r=script_args.peft_lora_r,
        lora_alpha=script_args.peft_lora_alpha,
        lora_dropout=0.05,
        target_modules = ["q_proj", "v_proj"]
    )

else:
    peft_config = None

from transformers import LlamaForCausalLM, LlamaTokenizer, get_linear_schedule_with_warmup, set_seed

llama_tokenizer = LlamaTokenizer.from_pretrained(script_args.model_name)
llama_tokenizer.padding_side = 'right'
llama_tokenizer.pad_token = llama_tokenizer.eos_token
print("training...")
# Step 5: Define the Trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=llama_tokenizer,
    args=training_args,
    max_seq_length=script_args.seq_length,
    train_dataset=dataset,
    dataset_text_field=script_args.dataset_text_field,
    peft_config=peft_config,
    # optim="adamw_torch_fused"
)
# import pdb; pdb.set_trace()

trainer.train()
print("saving...")
# Step 6: Save the model
# import pdb; pdb.set_trace()
trainer.save_model(script_args.output_dir)

