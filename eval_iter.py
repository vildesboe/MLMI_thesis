import numpy as np
import pandas as pd

# Some stats for files with max_loops larger than 1.

if __name__ == '__main__':
    folders1 = ['g4_iter/AE/']
    folders = folders1
    #folders = ['logs/finetunes_new/Vicuna/mine/', 'logs/oob_vicuna/A/']
    spec_names = ['i_', 'i2_']
    names = ["Chap", "Charl", "Chlo", "L", "Lam", "Mel", "Novak", "O", "P", "Palm"]
    
    for j in range(len(folders)):
        broken_princ_config = []
        all_numbers = []
        for t in range(len(spec_names)):
            for i in range(len(names)):
                file_name = folders[j] + spec_names[t] + names[i]
                file = open(file_name,mode='r')
                full_file = file.read()
                splits = full_file.split('yes-es this loop.\n')[:-1]
                file.close()
                tots = full_file.split('We did a total of')
                nr_loops = (int(tots[1][:3].strip()))
                yes_es = [int(s[-4:].replace('e', '').strip()) for s in splits]
                print(f"{names[i]:6s}: {yes_es}; {nr_loops}")
