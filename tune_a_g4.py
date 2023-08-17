import os
import sys
from typing import List

import torch
import transformers
from datasets import load_dataset

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

base_model= "lmsys/vicuna-13b-v1.3"
data_path = "/home/vsb29/rds/hpc-work/project/data/GPT4_full3.json"
output_dir = "./tunev_g4_E_t1"

# training hyperparams
batch_size = 32
micro_batch_size = 4
num_epochs = 5
learning_rate = 8e-4
cutoff_len = 2000
val_set_size = 50
# lora hyperparams
lora_r = 8
lora_alpha = 16
lora_dropout = 0.05
lora_target_modules = [
    "q_proj",
    "v_proj",
]
# llm hyperparams
train_on_inputs = False  # if False, masks out inputs in loss
add_eos_token = False
group_by_length = False  # faster, but produces an odd training loss curve

only_yesno = False
explain_and_answer = False
explain_only = False
if '_A_' in output_dir:
    only_yesno = True
elif '_AE_' in output_dir:
    explain_and_answer = True
elif '_E_' in output_dir:
    explain_only = True
else:
    print("NO VALID OUTPUT FOLDER NAME. This run will break.")

if int(os.environ.get("LOCAL_RANK", 0)) == 0:
    print(
        f"Training Alpaca-LoRA model with params:\n"
        f"base_model: {base_model}\n"
        f"data_path: {data_path}\n"
        f"output_dir: {output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"val_set_size: {val_set_size}\n"
        f"lora_r: {lora_r}\n"
        f"lora_alpha: {lora_alpha}\n"
        f"lora_dropout: {lora_dropout}\n"
        f"lora_target_modules: {lora_target_modules}\n"
        f"train_on_inputs: {train_on_inputs}\n"
        f"add_eos_token: {add_eos_token}\n"
        f"group_by_length: {group_by_length}\n"
    )
assert (
    base_model
), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
gradient_accumulation_steps = batch_size // micro_batch_size

device_map = "auto"
world_size = int(os.environ.get("WORLD_SIZE", 1))
print("world size", world_size)
ddp = world_size != 1
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    gradient_accumulation_steps = 4#gradient_accumulation_steps // world_size


quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=quant_config, device_map=device_map)

print("Q-Lora")
# model = AutoModelForCausalLM.from_pretrained(
#     base_model,
#     load_in_8bit=True,
#     torch_dtype=torch.float16,
#     device_map=device_map,
# )
model.gradient_checkpointing_enable() # Gradient cehckpointing, reduces memory req

tokenizer = AutoTokenizer.from_pretrained(base_model)

tokenizer.pad_token_id = (
    0  # unk. we want this to be different from the eos token
)
tokenizer.padding_side = "left"  # Allow batched inference

def tokenize(prompt, add_eos_token=True):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < cutoff_len
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result

def generate_prompt(only_explanation, explanation_and_answer, answer=None, input="", question=""):
    system_prompt="A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."

    yesno_A = " Please answer only 'yes' or 'no'."
    yesno_AE = " Start by thinking about the question step by step, and include relevant quotes from the input story and from the statement. Then always end by answering the question concisely with <yes> or <no> being the last word."
    yesno_E = " First, locate quotes from the input story and from the statement that may be relevant to the question. Please think step by step, but do NOT answer the question."
    diff = " Make sure to think carefully about what is said in the input story VS what is said in the statement."

    if answer:
        if only_yesno:
            q = question + yesno_A + diff
            msg = q + '\n' + input # q + (story & statement)
            return f"""{system_prompt} USER: {msg} ASSISTANT: {answer}""" # Only yes/no answer
        elif explain_and_answer:
            q = question + yesno_AE + diff
            msg = q + '\n' + input # q + (story & statement)
            return f"""{system_prompt} USER: {msg} ASSISTANT: {explanation_and_answer}""" # Explain why -> yes/no
        elif explain_only:
            q = question + yesno_E + diff
            msg = q + '\n' + input # q + (story & statement)
            return f"""{system_prompt} USER: {msg} ASSISTANT: {only_explanation}""" # Only discuss the question (hopefully no 'yes', / 'no')

    else:
        if only_yesno:
            q = question + yesno_A + diff
        elif explain_and_answer:
            q = question + yesno_AE + diff
        elif explain_only:
            q = question + yesno_E + diff
        msg = q + '\n' + input # q + (story & statement)
        return f"""{system_prompt} USER: {msg} ASSISTANT:"""

def generate_and_tokenize_prompt(data_point):
    # only_explanation, explanation_and_answer, answer, input, question
    full_prompt = generate_prompt(
        data_point["only_explanation"],
        data_point["explanation_and_answer"],
        data_point["answer"].strip(),
        data_point["input"],
        data_point["question"],
    )
    tokenized_full_prompt = tokenize(full_prompt)
    if not train_on_inputs:
        temp_point = data_point.copy()
        #temp_point["answer"] = ""
        user_prompt = generate_prompt(
            temp_point["only_explanation"],
            temp_point["explanation_and_answer"],
            answer=None,
            input=temp_point["input"],
            question=temp_point["question"],
        )

        tokenized_user_prompt = tokenize(
            user_prompt, add_eos_token=add_eos_token
        )
        user_prompt_len = len(tokenized_user_prompt["input_ids"])

        if add_eos_token:
            user_prompt_len -= 1

        tokenized_full_prompt["labels"] = [
            -100
        ] * user_prompt_len + tokenized_full_prompt["labels"][
            user_prompt_len:
        ]  # could be sped up, probably
    return tokenized_full_prompt

model = prepare_model_for_int8_training(model)

config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    target_modules=lora_target_modules,
    lora_dropout=lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)

if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()
else:
    def make_inputs_require_grad(module, input, output):
         output.requires_grad_(True)
    model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
#print("Less memory")

if data_path.endswith(".json") or data_path.endswith(".jsonl"):
    data = load_dataset("json", data_files=data_path)
else:
    data = load_dataset(data_path)

# if resume_from_checkpoint:
#     # Check the available weights and load them
#     checkpoint_name = os.path.join(
#         resume_from_checkpoint, "pytorch_model.bin"
#     )  # Full checkpoint
#     if not os.path.exists(checkpoint_name):
#         checkpoint_name = os.path.join(
#             resume_from_checkpoint, "adapter_model.bin"
#         )  # only LoRA model - LoRA config above has to fit
#         resume_from_checkpoint = (
#             False  # So the trainer won't try loading its state
#         )
#     # The two files above have a different name depending on how they were saved, but are actually the same.
#     if os.path.exists(checkpoint_name):
#         print(f"Restarting from {checkpoint_name}")
#         adapters_weights = torch.load(checkpoint_name)
#         set_peft_model_state_dict(model, adapters_weights)
#     else:
#         print(f"Checkpoint {checkpoint_name} not found")

model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

if val_set_size > 0:
    train_val = data["train"].train_test_split(
        test_size=val_set_size, shuffle=True, seed=42
    )
    train_data = (
        train_val["train"].shuffle().map(generate_and_tokenize_prompt, batched=False)
    )
    val_data = (
        train_val["test"].shuffle().map(generate_and_tokenize_prompt, batched=False)
    )
else:
    train_data = data["train"].shuffle().map(generate_and_tokenize_prompt, batched=False)
    val_data = None

print("About the data...")
print(val_data[0]["labels"][-5:])
print(val_data[0]["input_ids"][-5:])
print(train_data[0]["labels"][-5:])
print(train_data[0]["input_ids"][-5:])
#print(tokenizer.decode(train_data[0]["labels"][-5:])) # Need to remove the -100 to work


print("To training!")

if not ddp and torch.cuda.device_count() > 1:
    # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
    model.is_parallelizable = True
    model.model_parallel = True

trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=transformers.TrainingArguments(
        #per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=5,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        fp16=True,
        logging_steps=1,
        optim="adamw_torch",
        evaluation_strategy="steps" if val_set_size > 0 else "no",
        save_strategy="steps",
        #eval_steps=5 if val_set_size > 0 else None,
        save_steps=10,
        output_dir=output_dir,
        #save_total_limit=3,
        load_best_model_at_end=True if val_set_size > 0 else False,
        ddp_find_unused_parameters=False if ddp else None,
        group_by_length=group_by_length,
        report_to="none",
    ),
    data_collator=transformers.DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    ),
)
model.config.use_cache = False

old_state_dict = model.state_dict
model.state_dict = (
    lambda self, *_, **__: get_peft_model_state_dict(
        self, old_state_dict()
    )
).__get__(model, type(model))

if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)

#trainer.train(resume_from_checkpoint=resume_from_checkpoint)
trainer.train()

model.save_pretrained(output_dir)

print(
    "\n If there's a warning about missing keys above, please disregard :)"
)
print("Training done! :D")

import math
eval_results = trainer.evaluate()
print(f"Perplexity, {math.exp(eval_results['eval_loss']):.2f}")

