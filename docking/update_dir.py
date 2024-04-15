import pickle
import os

from rdkit.Chem.rdmolfiles import MolToPDBBlock, MolToPDBFile
from rdkit import Chem
from rdkit.Chem import MolFromSmiles, AddHs, AllChem
from rdkit import Geometry

import pandas as pd
import shutil
from tqdm import tqdm

def create_diffdock_complex(split='train'):
    with open(f'docking/DiffDock/data/cacheNew/davis_colabfold_model_davis_data_split_train_limit_20/ligand_positions_rank1.pkl', 'rb') as f:
        rank1_ligand_positions = pickle.load(f)

    dir_parent = f'data/benchmark/complex'

    dict_name_coord = rank1_ligand_positions

    for name in tqdm(dict_name_coord):
        try: 
            dir_path = os.path.join(dir_parent, name)
            # lig_mol2_path = os.path.join(dir_path,f'{pdbid}_ligand.mol2')
            pro_name, lig_name = name.split('_')[0], name.split('_')[1]
            lig_sdf_path = os.path.join(dir_path,f'{lig_name}_ligand.sdf')
            

            supplier = Chem.SDMolSupplier(lig_sdf_path, sanitize=True, removeHs=True)
            mol = supplier[0]
        
                
            if mol:
                coords = dict_name_coord[name]
                assert coords.shape[0] == mol.GetConformer().GetPositions().shape[0]
                for i in range(coords.shape[0]):
                    mol.GetConformer().SetAtomPosition(i, Geometry.Point3D(float(coords[i, 0]), float(coords[i, 1]), float(coords[i, 2])))
    
                mol.SetProp("_pdbid_", name)
    
                # lig_pdb_diffdock_path = os.path.join(dir_path, f'{name}_ligand_diffdock.pdb')
                lig_sdf_diffdock_path = os.path.join(dir_path, f'{lig_name}_ligand_diffdock.sdf')

                # MolToPDBFile(mol, lig_pdb_diffdock_path)
                
                writer = Chem.SDWriter(lig_sdf_diffdock_path)
                writer.write(mol)
                writer.close()
            else:
                print(f'cannot process {name}, bad sdf ligand')

        except Exception as e:
            print(f'cannot process {name}, error: {e}')

if __name__ == '__main__':
    create_diffdock_complex()