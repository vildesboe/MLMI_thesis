import numpy as np
import pandas as pd
import os


folders1 = ['initials/', 'DR/', 'g4/AR/', 'g4/A/', 'g4/AE/', 'g4/E/', 'g4_iter/AE/']
folders2 = ['logs/oob_vicuna/AR/', 'logs/oob_vicuna/A/', 'logs/oob_vicuna/AE/', 'logs/oob_vicuna/E/']
folders3 = ['logs/finetunes_new/Vicuna/mine/', 'logs/finetunes_new/Vicuna/mine_AR/', 'logs/finetunes_new/Vicuna/AR/', 'logs/finetunes_new/Vicuna/A/', 'logs/finetunes_new/Vicuna/AE/', 'logs/finetunes_new/Vicuna/E/', 'logs/finetunes_new/Vicuna/AE/']
n3s = ['g4/AE/', 'g4/A/', 'logs/oob_vicuna/A/', 'logs/finetunes_new/Vicuna/mine/', 'logs/finetunes_new/Vicuna/A/']
folders = folders1 + folders2 + folders3
folders = ['logs/oob_vicuna/A/']
folders = ['logs/finetunes_new/Vicuna/A/']
folders = [n3s[0]]
#folders = ['logs/finetunes_new/Vicuna/mine/', 'logs/oob_vicuna/A/']
eval_files = ['eval.txt', 'eval2.txt', 'eval3.txt', 'eval4.txt', 'eval5.txt']
names = ["Chap", "Charl", "Chlo", "L", "Lam", "Mel", "Novak", "O", "P", "Palm"]
plot = False

df_numbers = {}
nr_broken_principles = {'coherence':0, 'guilty plea':0, 'irrelevant':0, 'suffering':0, 'age':0, 'record':0, 'mental':0, 'explain':0, 'amends':0, 'improve + prevent':0, 'character':0, 'emotions':0, 'hallucination':0}
numbers_per_name = {"Chap":[], "Charl":[], "Chlo":[], "L":[], "Lam":[], "Mel":[], "Novak":[], "O":[], "P":[], "Palm":[]}
total_files = 0

for j in range(len(folders)):
        folder_occs = 0
        directory = folders[j]
        files_in_folder = 0
        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            # checking if it is a file
            if os.path.isfile(f) and 'eval' not in f:
                #print(f)
                total_files += 1
                files_in_folder += 1
                file = open(f,mode='r')
                full_file = file.read()
                file.close()

                occ = full_file.count("Unwanted format, we interpret as 'nothing wrong'")
                folder_occs += occ
        df_numbers[directory] = folder_occs/files_in_folder

print(df_numbers)