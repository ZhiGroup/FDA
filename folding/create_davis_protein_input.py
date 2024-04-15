#%%
import pandas as pd
import os
from Bio.PDB import PDBList
from Bio import SeqIO
import requests
import yaml
from tqdm import tqdm
#%%
def download_file(url, filename):
    response = requests.get(url)
    response.raise_for_status()  # Ensure we got an OK response

    with open(filename, 'wb') as f:
        f.write(response.content)

def read_fasta_file(filename):
    seq = ''
    for i, record in enumerate(SeqIO.parse(filename, "fasta")):
        if i > 0:
            seq += ':'
            seq += record.seq
        else:
            seq += record.seq
    return str(seq)


def read_yaml_file(filename):
    with open(filename, 'r') as file:
        data = yaml.safe_load(file)
    return data


dir = 'folding'

dict_name_pdbid = read_yaml_file(os.path.join(dir, 'davis_protein2pdb.yaml'))

ls_seq = []
ls_name = []

for name, pdbid in tqdm(dict_name_pdbid.items()):
    pdbid = pdbid.split('.')[0]
    download_file(f'https://www.rcsb.org/fasta/entry/{pdbid}', f'{pdbid}.fasta')
    ls_seq.append(read_fasta_file(f'{pdbid}.fasta'))
    ls_name.append(name)
    os.remove(f'{pdbid}.fasta')

df = pd.DataFrame({'id': ls_name, 'sequence': ls_seq})
df.to_csv(os.path.join(dir, 'input/davis_protein.csv'), index=False)