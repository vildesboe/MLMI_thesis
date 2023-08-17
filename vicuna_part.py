# Huggingface API
import os
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from fastchat.model import load_model#, get_conversation_template, add_model_args
from vicuna_prompts import get_conv_template

if torch.cuda.is_available():
    device = "cuda"
else:
    print("CUDA not available. Attempting use of CPU.")
    device = "cpu"

model_name = "vicuna_v1.1"
model_path = "lmsys/vicuna-13b-v1.3"
model, tokenizer = load_model("lmsys/vicuna-13b-v1.3", device, 1, 35, True, True)

model.eval()

def evaluate(instruction,
    input="",
    temperature=0.5,
    top_p=0.75,
    top_k=50,
    num_beams=4,
    max_new_tokens=2,
    shots=0,
    **kwargs,):
    msg = instruction + "\n" + input
    if shots == 2:
        print("Using 2-shot vicuna prompt")
        conv = get_conv_template("vicuna_2shot")
    elif shots ==3.5:
        conv = get_conv_template("turbo3.5")
    else:
        conv = get_conv_template("vicuna_v1.1")

    conv.append_message(conv.roles[0], msg)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer([prompt]).input_ids
    output_ids = model.generate(
        torch.as_tensor(input_ids).cuda(),
        do_sample=True,
        temperature=temperature,
        top_p=0.75,
        repetition_penalty=1,
        max_new_tokens=max_new_tokens,
    )

    if model.config.is_encoder_decoder:
        output_ids = output_ids[0]
    else:
        output_ids = output_ids[0][len(input_ids[0]) :]
    outputs = tokenizer.decode(
        output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
    )
    return outputs
