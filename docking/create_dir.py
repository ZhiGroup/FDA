import pandas as pd
import os
import glob
import shutil
from tqdm import tqdm
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
     
    parser.add_argument('--data_dir', type=str, default='data/benchmark/',
                        help='the name of complex dir')

    parser.add_argument('--complex_dir_name', type=str, default='complex',
                        help='the name of complex dir')

    parser.add_argument('--protein_dir_name', type=str, default='davis_colabfold_protein',
                        help='the name of protein dir')
    
    parser.add_argument('--ligand_dir_name', type=str, default='davis_ligand',
                        help='the name of protein dir')
    
    parser.add_argument('--data_csv', type=str, default='davis_data.tsv',
                        help='the name of data csv file')

    args = parser.parse_args()
    return args


args = parse_arguments()

data_complex_dir = os.path.join(args.data_dir, f'{args.complex_dir_name}') 
davis_data = pd.read_csv(os.path.join(args.data_dir, f'{args.data_csv}'), sep='\t')


if not os.path.exists(data_complex_dir):
    os.makedirs(data_complex_dir)

for i, row in tqdm(davis_data.iterrows()):
    protein_name, ligand_name = row['protein'], row['drug']
    name = f'{protein_name}_{ligand_name}'
    
    if not os.path.exists(f'{os.path.join(data_complex_dir, name)}'):
        os.makedirs(f'{os.path.join(data_complex_dir, name)}')
    
    if not os.path.exists(f'{os.path.join(data_complex_dir, name, f"{protein_name}_protein_processed.pdb")}'):
        try:
            protein_path = f'{os.path.join(args.data_dir, args.protein_dir_name, protein_name)}.pdb'
            shutil.copy(protein_path, f'{os.path.join(data_complex_dir, name, f"{protein_name}_protein_processed.pdb")}')
        except Exception as e:
            print(e, f'there is no protein file for {protein_name}')
            # shutil.rmtree(f'data/kdbnet_colabfold_diffdock/{name}')
    
    shutil.copy(f'{os.path.join(args.data_dir, args.ligand_dir_name, f"{ligand_name}.sdf")}', 
                f'{os.path.join(data_complex_dir, name, f"{ligand_name}_ligand.sdf")}')