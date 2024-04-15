#%%
import pandas as pd
import os
import glob

for file_name in os.listdir('data/davis'):
    if '_' not in file_name:
        print(file_name)

# %%
ori_data = pd.read_csv('../../data/benchmark/davis_data.tsv', sep='\t')
ori_data = [f'{protein}_{ligand}' for protein, ligand in zip(ori_data['protein'], ori_data['drug'])]
# %%
result = set(os.listdir('data/davis')) - set(set(os.listdir('data/davis')) & set(ori_data))
# %%
import glob
fail = 0
complex_pth = os.listdir('../../data/benchmark/complex')

for folder_path in complex_pth:
    folder_path = os.path.join('../../data/benchmark/complex', folder_path)
    pyg_files = glob.glob(folder_path + '/*.pyg')

    if len(pyg_files) > 0:
        pass
    else:
        fail += 1
print(fail)

# %%
