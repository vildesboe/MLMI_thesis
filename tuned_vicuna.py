"""
FILE DESCRIPTION:

Inference on a model based on Vicuna13, but with new weights from LoRA.
Loads the model (Vicuna13 & LORA), and the method 'evaluate' is called to do actual inference.
"""

import os
import sys

import gradio as gr
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig#, LlamaForCausalLM, LlamaTokenizer
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# Change this to change the LORA-part (which fine-tune to use)
base_model = "lmsys/vicuna-13b-v1.3"
lora_weights = "./tunev_g4_A_2"
print(f"Using {lora_weights};")

# Start loading the model
if torch.cuda.is_available():
    device = "cuda"
else:
    print("CUDA not available. Attempting use of CPU.")
    device = "cpu"
load_8bit = True

tokenizer = AutoTokenizer.from_pretrained(base_model)
if device == "cuda":
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(
        model,
        lora_weights,
        torch_dtype=torch.float16,
    )

model.eval()
if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)

def evaluate(
    instruction,
    input=None,
    temperature=0.5,
    top_p=0.75,
    top_k=50,
    num_beams=4,
    max_new_tokens=2,
    shots=0,
    **kwargs,
):
    
    def generate_vicuna_prompt(instruction, input=""):
        system_prompt="A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
        if input!="":
            msg = instruction + "\n" + input
            return f"""{system_prompt} USER: {msg} ASSISTANT:"""
        else:
            return f"""{system_prompt} USER: {instruction} ASSISTANT:"""
        
    def get_response(output):
        #print("output type:", type(output))
        out = output.split("<\s>")[0].split("ASSISTANT:")
        if len(out)>2:
            print("WE HAVE AN UNWANTED RESPONSE W 2 ASSISTANTS!:", print(out[1:]))
        return out[-1].replace("</s>", "").strip()
        
    prompt = generate_vicuna_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
        **kwargs,
    )

    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    #print("output in tuned_vicuna: ", output) # -> full output w description, instruction, input, response
    return get_response(output)