import numpy as np
import os
import pandas as pd
import re

# analysis based on the eval.txt files

def metrics(splits):
    names = [l.split('\n')[0] for l in splits]
    rest = [l.split('\n')[-1] for l in splits]

    # Which principle is most difficult to satisfy?
    pens_for = [l.split(':')[0] for l in rest]
    pens_for = [l.strip("][").replace("'", "").split(', ') for l in pens_for]
    #print(pens_for)
    nr_brokens_princ = {'coherence':0, 'guilty plea':0, 'irrelevant':0, 'suffering':0, 'age':0, 'record':0, 'mental':0, 'explain':0, 'amends':0, 'improve + prevent':0, 'character':0, 'emotions':0, 'hallucination':0}
    for s in range(len(pens_for)):
        story = pens_for[s]
        for p in range(len(story)):
            if story[p] != '':
                nr_brokens_princ[story[p]] += 1
    #print(nr_brokens_princ)

    # Number of proken penalties
    numbers = [int(l.split(': ')[1]) for l in rest]
    #print(numbers)

    return nr_brokens_princ, numbers


if __name__ == '__main__':
    folders1 = ['initials/', 'DR/', 'g4/AR/', 'g4/A/', 'g4/AE/', 'g4/E/', 'g4_iter/AE/']
    folders2 = ['logs/oob_vicuna/AR/', 'logs/oob_vicuna/A/', 'logs/oob_vicuna/AE/', 'logs/oob_vicuna/E/']
    folders3 = ['logs/finetunes_new/Vicuna/mine/', 'logs/finetunes_new/Vicuna/mine_AR/', 'logs/finetunes_new/Vicuna/AR/', 'logs/finetunes_new/Vicuna/A/', 'logs/finetunes_new/Vicuna/AE/', 'logs/finetunes_new/Vicuna/E/']
    n3s = ['initials/', 'DR/', 'g4/AE/', 'g4/A/', 'logs/oob_vicuna/A/', 'logs/finetunes_new/Vicuna/mine/', 'logs/finetunes_new/Vicuna/A/']
    folders = folders1 + folders2 + folders3
    folders = [n3s[-1]]
    #folders = ['logs/finetunes_new/Vicuna/AE/']
    #folders = ['logs/finetunes_new/Vicuna/mine/', 'logs/oob_vicuna/A/']
    eval_files = ['eval.txt', 'eval2.txt', 'eval3.txt', 'eval4.txt', 'eval5.txt']
    spec_names = ['m_', 'm2_']
    names = ["Chap", "Charl", "Chlo", "L", "Lam", "Mel", "Novak", "O", "P", "Palm"]
    plot = False

    df_numbers = pd.DataFrame()
    df_stats = pd.DataFrame()
    nr_broken_principles = {'coherence':0, 'guilty plea':0, 'irrelevant':0, 'suffering':0, 'age':0, 'record':0, 'mental':0, 'explain':0, 'amends':0, 'improve + prevent':0, 'character':0, 'emotions':0, 'hallucination':0}
    numbers_per_name = {"Chap":[], "Charl":[], "Chlo":[], "L":[], "Lam":[], "Mel":[], "Novak":[], "O":[], "P":[], "Palm":[]}
    total_files = 0
    
    for j in range(len(folders)):
        broken_princ_config = []
        all_numbers = []
        files_in_config = 0
        for i in range(len(eval_files)):
            file_name = folders[j] + eval_files[i]
            print(file_name)
            if os.path.isfile(file_name):
                total_files += 1
                files_in_config += 1
                file = open(file_name,mode='r')
                full_file = file.read()
                splits = full_file.replace('#We have 10 files\n', '').split('_ ')
                if len(splits)<=1:
                    splits = full_file.replace('#We have 10 files\n', '').split('/ ')
                file.close()
                splits = splits[1:]
                splits = [l.split('\n\n')[0] for l in splits]
                broken_princ, numbers = metrics(splits)
                broken_princ_config.append(broken_princ)
                all_numbers.append(numbers)
                for n in range(len(numbers)):
                    numbers_per_name[names[n]].append(numbers[n])

        numbers_flat = np.array(all_numbers.copy()).flatten()
        full_dict_config = {key: broken_princ_config[0].get(key) + broken_princ_config[1].get(key) for key in set(broken_princ_config[0])}
        nr_broken_principles =  {key: full_dict_config.get(key) + nr_broken_principles.get(key) for key in set(nr_broken_principles)}

        config_name = folders[j].split('/')
        config_name = '/'.join(config_name[-3:])
        df_stats[config_name] = [np.mean(numbers_flat), np.std(numbers_flat)/np.sqrt(files_in_config*10), np.min(numbers_flat), np.max(numbers_flat)]
        df_numbers[config_name] = [all_numbers[0][e] + all_numbers[1][e] for e in range(len(all_numbers[0]))]

        if len(folders)==1 and plot:
            import matplotlib.pyplot as plt
            import seaborn as sns
            hist, bins = np.histogram(numbers_flat, bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            plt.hist(x=numbers_flat, edgecolor='k', bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            ticks = [patch + 0.5 for patch in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
            ticklabels = [i for i in range(10)]
            plt.xticks(ticks, ticklabels)
            plt.title(folders[0])
            plt.savefig('dist3.png')
            print(numbers_flat)
            print(hist)
            print(bins)


    df_numbers.index = names
    df_stats.index = ['Mean', 'StdE', 'min', 'max']

    # Update standard deviation to standard error
    #df_stats.loc['StdE'] = df_stats.loc['StdE']/np.sqrt(total_files)


    # Adding stats over each input story
    avgs = []
    stde = []
    maxx = []
    minn = []
    for ni in range(len(names)):
        numbers_list = numbers_per_name[names[ni]]
        avgs.append(np.mean(numbers_list))
        stde.append(np.std(numbers_list)/np.sqrt(total_files))
        maxx.append(np.max(numbers_list))
        minn.append(np.min(numbers_list))

    df_numbers['Average'] = avgs
    df_numbers['StdE'] = stde
    df_numbers['min'] = minn
    df_numbers['max'] = maxx
    # Adding stats over each config
    df_numbers = pd.concat([df_numbers, df_stats], axis=0)
    
    print(df_numbers)
    #print(df_stats)
    #print(nr_broken_principles)

    print(f"# configs considered: {len(folders)}")
    #total_files = len(folders)*len(eval_files)*10
    print(f"# files considered: {total_files}") # Number of statements
    print(f"# statements considered: {total_files*10}") 

    #print(numbers_per_name)

    #with pd.ExcelWriter('analysis_evals.xlsx', mode='w') as writer: 
    #    df_numbers.to_excel(writer, sheet_name='Numbers')

    # import seaborn as sns
    # import matplotlib.pyplot as plt
    # sns.set()
    # nr_broken_principles.update((x, 100*y/total_files) for x, y in nr_broken_principles.items())
    # princ_df = pd.DataFrame(nr_broken_principles.items())
    # #print(princ_df.head())
    
    # # Broken principles plot
    # ax = sns.barplot(x=0, y=1, data=princ_df)
    # ax.set(xlabel='Principle', ylabel='% of statements w broken principle', title='Frequency of breaking the principles')
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=60)
    # plt.tight_layout()
    # plt.savefig('fig2.png', dpi=800)

    #print(numbers_per_name)