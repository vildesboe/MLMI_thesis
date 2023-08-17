# MLMI_thesis
Codebase for the thesis "LLM self-critique for task-oriented text generation". Please note that the code has not been cleaned, or made for others to look at, but feel free to look around!

Please note: this repo does not contain any keys for OpenAI that you would need to run parts of this code! Please insert your own key if you want to try out the code, or contact me.

### PLEASE NOTE: Files that are not my own work:
The files for fine-tuning: tune_a_v.py and tune_a_g4 are based on fine-tuning files from the Alpaca repository and from the Vicuna repository.

Link to Alpaca repo: https://github.com/tloen/alpaca-lora

Link to Vicuna repo: https://github.com/lm-sys/FastChat/tree/a47b8f9e93c8b5a85e81d1ae33e3a1106d8cdf80

### Short overview of most important contents:
The files beginning with combined_ contain code for running methodologies; eg initial, DR, etc..

Most folders contain examples of statements produced with the different setups (this should be relatively clear by the folder names).

The data folder contains my training data

The folders starting with tuned_g4 // tuned_my contain the adapters for the fine-tuned versions of Vicuna
