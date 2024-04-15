import pandas as pd
import os
import glob
import shutil
from tqdm import tqdm

data_dir = 'data/benchmark/'
data_complex_dir = os.path.join(data_dir, 'complex') 
davis_data = pd.read_csv(os.path.join(data_dir, 'davis_data.tsv'), sep='\t')


if not os.path.exists(data_complex_dir):
    os.makedirs(data_complex_dir)

for i, row in tqdm(davis_data.iterrows()):
    protein_name, ligand_name = row['protein'], row['drug']
    name = f'{protein_name}_{ligand_name}'
    
    if not os.path.exists(f'{os.path.join(data_complex_dir, name)}'):
        os.makedirs(f'{os.path.join(data_complex_dir, name)}')
    
    if not os.path.exists(f'{os.path.join(data_complex_dir, name, f"{protein_name}_protein_processed.pdb")}'):
        try:
            protein_path = f'{os.path.join(data_dir, "davis_colabfold_protein", protein_name)}.pdb'
            shutil.copy(protein_path, f'{os.path.join(data_complex_dir, name, f"{protein_name}_protein_processed.pdb")}')
        except Exception as e:
            print(e, f'there is no protein file for {protein_name}')
            # shutil.rmtree(f'data/kdbnet_colabfold_diffdock/{name}')
    
    shutil.copy(f'{os.path.join(data_dir, "davis_ligand", f"{ligand_name}.sdf")}', 
                f'{os.path.join(data_complex_dir, name, f"{ligand_name}_ligand.sdf")}')