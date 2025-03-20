#%%
import pickle
import os

from rdkit.Chem.rdmolfiles import MolToPDBBlock, MolToPDBFile
from rdkit import Chem
from rdkit.Chem import MolFromSmiles, AddHs, AllChem
from rdkit import Geometry

import pandas as pd
import shutil
from tqdm import tqdm
import argparse
import pickle
import copy


def parse_args():
    parser = argparse.ArgumentParser(description='Description of your script')
    
    # Add arguments here
    parser.add_argument('--diffdock_ligand_path', type=str, default='docking/DiffDock/data/cacheNew/davis_colabfold_model_davis_data_split_train_limit_0/ligand_positions_rank1.pkl', help='the path of diffdock ligand positions')
    parser.add_argument('--complex_path', type=str, default='data/benchmark/davis_complex_colabfold_diffdock', help='the path of diffdock protein positions')
    parser.add_argument('--top_n', type=str, default=1, help='save top n ligands')
    args = parser.parse_args()
    return args

def create_diffdock_complex(args):
    with open(args.diffdock_ligand_path, 'rb') as f:
        rank1_ligand_positions = pickle.load(f)

    all_ligands_path = os.path.join(os.path.dirname(args.diffdock_ligand_path), 'ligand_positions.pkl')
    with open(all_ligands_path, 'rb') as f:
        all_ligand_positions = pickle.load(f)

    dir_parent = args.complex_path

    dict_name_coord_rank1 = rank1_ligand_positions
    dict_name_coord_all = all_ligand_positions

    for name in tqdm(dict_name_coord_rank1):
        try: 
            dir_path = os.path.join(dir_parent, name)
            # lig_mol2_path = os.path.join(dir_path,f'{pdbid}_ligand.mol2')
            lig_name = name.split('_')[-1]
            lig_sdf_path = os.path.join(dir_path,f'{lig_name}_ligand.sdf')
            lig_sdf_diffdock_path = os.path.join(dir_path, f'{lig_name}_ligand_diffdock.sdf')

            supplier = Chem.SDMolSupplier(lig_sdf_path, sanitize=True, removeHs=True)
            mol = supplier[0]
            mol_ori = copy.deepcopy(mol)
            

            if not os.path.exists(lig_sdf_diffdock_path):
                    
                if mol:
                    coords = dict_name_coord_rank1[name]
                    assert coords.shape[0] == mol.GetConformer().GetPositions().shape[0]
                    for i in range(coords.shape[0]):
                        mol.GetConformer().SetAtomPosition(i, Geometry.Point3D(float(coords[i, 0]), float(coords[i, 1]), float(coords[i, 2])))
        
                    mol.SetProp("_complex_", name)
                    
                    writer = Chem.SDWriter(lig_sdf_diffdock_path)
                    writer.write(mol)
                    writer.close()
                else:
                    print(f'cannot process {name}, bad sdf ligand')
            
            # here for save top n ligands
            if int(args.top_n) > 1:

                for i in range(int(args.top_n)):
                    lig_sdf_diffdock_path = os.path.join(dir_path, f'{lig_name}_ligand_diffdock_{i+1}.sdf')
                    if os.path.exists(lig_sdf_diffdock_path):
                        continue
                    mol = copy.deepcopy(mol_ori)
                    if mol:
                        coords = dict_name_coord_all[name][i]
                        assert coords.shape[0] == mol.GetConformer().GetPositions().shape[0]
                        for j in range(coords.shape[0]):
                            mol.GetConformer().SetAtomPosition(j, Geometry.Point3D(float(coords[j, 0]), float(coords[j, 1]), float(coords[j, 2])))
                        mol.SetProp("_complex_", name)
                        writer = Chem.SDWriter(lig_sdf_diffdock_path)
                        writer.write(mol)
                        writer.close()
                    else:
                        print(f'cannot process {name}, bad sdf ligand')

        except Exception as e:
            print(f'cannot process {name}, error: {e}')

if __name__ == '__main__':
    args = parse_args()
    create_diffdock_complex(args)