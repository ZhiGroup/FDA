# %%
import os
import pickle
from rdkit import Chem
import pandas as pd
from tqdm import tqdm
import pymol
from rdkit import RDLogger
import argparse

RDLogger.DisableLog('rdApp.*')

def parse_args():
    parser = argparse.ArgumentParser(description='Description of your script')
    
    # Add arguments here
    parser.add_argument('--data_df', type=str, default='data/benchmark/davis_data.tsv', help='data of protein and ligand')
    parser.add_argument('--complex_path', type=str, default='data/benchmark/davis_complex_colabfold_diffdock', help='the path of diffdock protein positions')
    parser.add_argument('--distance', type=int, default=5, help='distance to generate pocket')
    parser.add_argument('--input_ligand_format', type=str, default='sdf', help='input ligand format')
    parser.add_argument('--top_n', type=int, default=1, help='preprocess top n ligands')
    args = parser.parse_args()
    return args


def generate_pocket(data_dir, top_n, distance=5):
    ls_name = os.listdir(data_dir)
    for name in tqdm(ls_name):
        try: 
            if name == '.DS_Store':
                continue
            complex_dir = os.path.join(data_dir, name)
            lig_name = name.split('_')[-1]
            pro_name = name.replace(f'_{lig_name}', '')

            ligand_path = os.path.join(complex_dir, f"{lig_name}_ligand_diffdock.sdf")
            protein_path= os.path.join(complex_dir, f"{pro_name}_protein_processed.pdb")

            if not os.path.exists(os.path.join(complex_dir, f'Pocket_{distance}A.pdb')):
                pymol.cmd.load(protein_path)
                pymol.cmd.remove('resn HOH')
                pymol.cmd.load(ligand_path)
                pymol.cmd.remove('hydrogens')
                pymol.cmd.select('Pocket', f'byres {lig_name}_ligand_diffdock around {distance}')
                pymol.cmd.save(os.path.join(complex_dir, f'Pocket_{distance}A.pdb'), 'Pocket')
                pymol.cmd.delete('all')

            if top_n > 1:
                for i in range(top_n):
                    if not os.path.exists(os.path.join(complex_dir, f'Pocket_{distance}A_{i+1}.pdb')):
                        ligand_path = os.path.join(complex_dir, f"{lig_name}_ligand_diffdock_{i+1}.sdf")
                        pymol.cmd.load(protein_path)
                        pymol.cmd.remove('resn HOH')
                        pymol.cmd.load(ligand_path)
                        pymol.cmd.remove('hydrogens')
                        pymol.cmd.select('Pocket', f'byres {lig_name}_ligand_diffdock_{i+1} around {distance}')
                        pymol.cmd.save(os.path.join(complex_dir, f'Pocket_{distance}A_{i+1}.pdb'), 'Pocket')
                        pymol.cmd.delete('all')

        except Exception as e:
            print(f'cannot generate pocket for {name}, due to {e}')

def generate_complex(data_dir, data_df, top_n, distance=5, input_ligand_format='sdf'):
    data_df = pd.read_csv(data_df, sep='\t')
    pbar = tqdm(total=len(data_df))
    for i, row in data_df.iterrows():
        try:
            pro_name, lig_name = row['protein'], row['drug']
            name = f'{pro_name}_{lig_name}'
            complex_dir = os.path.join(data_dir, name)
            pocket_path = os.path.join(data_dir, name, f'Pocket_{distance}A.pdb')
            save_path = os.path.join(complex_dir, f"{name}_{distance}A.rdkit")

            if not os.path.exists(save_path):
         
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

    
 

        except Exception as e:
            print(f'cannot generate complex for {name}, due to {e}')

        
        # here is for saving top n ligands
        if top_n > 1:
            for i in range(top_n):
                try:
                    pro_name, lig_name = row['protein'], row['drug']
                    name = f'{pro_name}_{lig_name}'
                    complex_dir = os.path.join(data_dir, name)
                    pocket_path = os.path.join(data_dir, name, f'Pocket_{distance}A_{i+1}.pdb')
                    save_path = os.path.join(complex_dir, f"{name}_{distance}A_{i+1}.rdkit")

                    if not os.path.exists(save_path):
    
                        if input_ligand_format != 'pdb':
                            ligand_input_path = os.path.join(data_dir, name, f'{lig_name}_ligand_diffdock_{i+1}.{input_ligand_format}')
                            ligand_path = ligand_input_path.replace(f".{input_ligand_format}", ".pdb")
                            os.chdir(complex_dir)
                            os.system(f'obabel {os.path.basename(ligand_input_path)} -O {os.path.basename(ligand_path)} -d')
                            os.chdir('../../../../')
                        else:
                            ligand_path = os.path.join(data_dir, name, f'{lig_name}_ligand_diffdock_{i+1}.pdb')

            
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

                except Exception as e:
                    print(f'cannot generate complex for ligand {i+1} of {name}, due to {e}')
        
        pbar.update(1)
        
        



if __name__ == '__main__':
    args = parse_args()
    ## generate pocket within 5 Ångström around ligand 
    generate_pocket(data_dir=args.complex_path, top_n=args.top_n, distance=args.distance)
    generate_complex(data_dir=args.complex_path, data_df=args.data_df, top_n=args.top_n, distance=args.distance, input_ligand_format=args.input_ligand_format)
    