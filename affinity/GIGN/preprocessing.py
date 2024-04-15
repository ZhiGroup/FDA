# %%
import os
import pickle
from rdkit import Chem
import pandas as pd
from tqdm import tqdm
import pymol
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# %%
def generate_pocket(data_dir, distance=5):
    ls_name = os.listdir(data_dir)
    for name in tqdm(ls_name):
        try: 
            if name == '.DS_Store':
                continue
            complex_dir = os.path.join(data_dir, name)
            pro_name, lig_name = name.split('_')[0], name.split('_')[1]

            ligand_path = os.path.join(complex_dir, f"{lig_name}_ligand_diffdock.sdf")
            protein_path= os.path.join(complex_dir, f"{pro_name}_protein_processed.pdb")

            # if os.path.exists(os.path.join(complex_dir, f'Pocket_{distance}A.pdb')):
            #     continue

            pymol.cmd.load(protein_path)
            pymol.cmd.remove('resn HOH')
            pymol.cmd.load(ligand_path)
            pymol.cmd.remove('hydrogens')
            pymol.cmd.select('Pocket', f'byres {lig_name}_ligand_diffdock around {distance}')
            pymol.cmd.save(os.path.join(complex_dir, f'Pocket_{distance}A.pdb'), 'Pocket')
            pymol.cmd.delete('all')
        except Exception as e:
            print(f'cannot generate pocket for {name}, due to {e}')

def generate_complex(data_dir, data_df, distance=5, input_ligand_format='sdf'):
    pbar = tqdm(total=len(data_df))
    for i, row in data_df.iterrows():
        try:
            pro_name, lig_name = row['protein'], row['drug']
            name = f'{pro_name}_{lig_name}'
            complex_dir = os.path.join(data_dir, name)
            pocket_path = os.path.join(data_dir, name, f'Pocket_{distance}A.pdb')
            save_path = os.path.join(complex_dir, f"{name}_{distance}A.rdkit")

            if os.path.exists(save_path):
                pbar.update(1)
                continue

            if input_ligand_format != 'pdb':
                ligand_input_path = os.path.join(data_dir, name, f'{lig_name}_ligand_diffdock.{input_ligand_format}')
                ligand_path = ligand_input_path.replace(f".{input_ligand_format}", ".pdb")
                os.chdir(complex_dir)
                os.system(f'obabel {os.path.basename(ligand_input_path)} -O {os.path.basename(ligand_path)} -d')
                os.chdir('../../../../')
            else:
                ligand_path = os.path.join(data_dir, name, f'{lig_name}_ligand_diffdock.pdb')

   
            ligand = Chem.MolFromPDBFile(ligand_path, removeHs=True)
            if ligand == None:
                ligand = Chem.SDMolSupplier(ligand_input_path)[0]

            if ligand == None:    
                print(f"Unable to process ligand of {name}, as {ligand_path} is problematic.")
                continue

            pocket = Chem.MolFromPDBFile(pocket_path, removeHs=True)
            if pocket == None:
                print(f"Unable to process protein of {name}, as {pocket_path} is problematic.")
                continue

            complex = (ligand, pocket)
            with open(save_path, 'wb') as f:
                pickle.dump(complex, f)

            pbar.update(1)

        except Exception as e:
            print(f'cannot generate complex for {name}, due to {e}')

if __name__ == '__main__':
    distance = 5
    input_ligand_format = 'sdf'
    data_dir = 'data/benchmark/complex'
    data_df = pd.read_csv('data/benchmark/davis_data.tsv', sep='\t')
    
    ## generate pocket within 5 Ångström around ligand 
    generate_pocket(data_dir=data_dir, distance=distance)
    generate_complex(data_dir, data_df, distance=distance, input_ligand_format=input_ligand_format)
    
# %%
