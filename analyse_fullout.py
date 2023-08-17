import numpy as np
import regex as re
import os

def find_nr_rewrites(file):
    relevant_text = file.read().split("yes-es this loop.")[0][-5:]
    nr = re.sub(r'[^\d]+', '', relevant_text)
    return nr


if __name__ == '__main__':
    folder = 'logs/oob_vicuna/A'
    #folder = 'g4/AE'
    folder = 'logs/finetunes_new/Vicuna/A'
    
    #spec_names = ['AE_', 'AE2_', 'AE3_', 'AE4_', 'AE5_']
    spec_names = ['A_', 'A2_', 'A3_', 'A4_', 'A5_']
    #spec_names = ['g4a_', 'g4a2_', 'g4a3_', 'A4_', 'A5_']
    #spec_names = ['g4ae_', 'g4ae2_']
    #spec_names = ['AE_', 'AE2_']
    names = ["Chap", "Charl", "Chlo", "L", "Lam", "Mel", "Novak", "O", "P", "Palm"]
    
    revisions = []
    for j in range(len(spec_names)):
        for i in range(len(names)):
            file_name = folder + '/' + spec_names[j] + names[i]
            if os.path.isfile(file_name):
                file = open(file_name,mode='r')
                nr_rewrites = int(find_nr_rewrites(file))
                file.close()
                #print(f"{names[i]}: {nr_rewrites}")
                revisions.append(nr_rewrites)

    print(folder)
    print(f"avg: {np.mean(revisions)}, std: {np.std(revisions)}")