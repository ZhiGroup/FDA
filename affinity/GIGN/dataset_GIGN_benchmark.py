# %%
import os
import pandas as pd
import numpy as np
import pickle
from scipy.spatial import distance_matrix
import multiprocessing
from itertools import repeat
import networkx as nx
import torch 
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit import RDLogger
from rdkit import Chem
from torch_geometric.data import Batch, Data
import warnings
from kdbnet.dta import create_fold, create_fold_setting_cold, create_full_ood_set, create_seq_identity_fold
import argparse

RDLogger.DisableLog('rdApp.*')
np.set_printoptions(threshold=np.inf)
warnings.filterwarnings('ignore')



def one_of_k_encoding(k, possible_values):
    if k not in possible_values:
        raise ValueError(f"{k} is not a valid value in {possible_values}")
    return [k == e for e in possible_values]


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(mol, graph, atom_symbols=['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I'], explicit_H=True):

    for atom in mol.GetAtoms():
        results = one_of_k_encoding_unk(atom.GetSymbol(), atom_symbols + ['Unknown']) + \
                one_of_k_encoding_unk(atom.GetDegree(),[0, 1, 2, 3, 4, 5, 6]) + \
                one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
                one_of_k_encoding_unk(atom.GetHybridization(), [
                    Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                    Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                        SP3D, Chem.rdchem.HybridizationType.SP3D2
                    ]) + [atom.GetIsAromatic()]
        # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
        if explicit_H:
            results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                    [0, 1, 2, 3, 4])

        atom_feats = np.array(results).astype(np.float32)

        graph.add_node(atom.GetIdx(), feats=torch.from_numpy(atom_feats))

def get_edge_index(mol, graph):
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        graph.add_edge(i, j)

def mol2graph(mol):
    graph = nx.Graph()
    atom_features(mol, graph)
    get_edge_index(mol, graph)

    graph = graph.to_directed()
    x = torch.stack([feats['feats'] for n, feats in graph.nodes(data=True)])
    edge_index = torch.stack([torch.LongTensor((u, v)) for u, v in graph.edges(data=False)]).T

    return x, edge_index

def inter_graph(ligand, pocket, dis_threshold = 5.):
    atom_num_l = ligand.GetNumAtoms()
    atom_num_p = pocket.GetNumAtoms()

    graph_inter = nx.Graph()
    pos_l = ligand.GetConformers()[0].GetPositions()
    pos_p = pocket.GetConformers()[0].GetPositions()
    dis_matrix = distance_matrix(pos_l, pos_p)
    node_idx = np.where(dis_matrix < dis_threshold)
    for i, j in zip(node_idx[0], node_idx[1]):
        graph_inter.add_edge(i, j+atom_num_l) 

    graph_inter = graph_inter.to_directed()
    edge_index_inter = torch.stack([torch.LongTensor((u, v)) for u, v in graph_inter.edges(data=False)]).T

    return edge_index_inter

# %%
def mols2graphs(complex_path, label, save_path, dis_threshold=5.):
    if not os.path.exists(complex_path):
        print(f"{complex_path} does not exist.")
        return
    # if os.path.exists(save_path):
    #     return
    try: 
        with open(complex_path, 'rb') as f:
            ligand, pocket = pickle.load(f)

        atom_num_l = ligand.GetNumAtoms()
        atom_num_p = pocket.GetNumAtoms()

        pos_l = torch.FloatTensor(ligand.GetConformers()[0].GetPositions())
        pos_p = torch.FloatTensor(pocket.GetConformers()[0].GetPositions())
        x_l, edge_index_l = mol2graph(ligand)
        x_p, edge_index_p = mol2graph(pocket)
        x = torch.cat([x_l, x_p], dim=0)
        edge_index_intra = torch.cat([edge_index_l, edge_index_p+atom_num_l], dim=-1)
        edge_index_inter = inter_graph(ligand, pocket, dis_threshold=dis_threshold)
        y = torch.FloatTensor([label])
        pos = torch.concat([pos_l, pos_p], dim=0)
        split = torch.cat([torch.zeros((atom_num_l, )), torch.ones((atom_num_p,))], dim=0)
        
        data = Data(x=x, edge_index_intra=edge_index_intra, edge_index_inter=edge_index_inter, y=y, pos=pos, split=split)

        torch.save(data, save_path)
    # return data
    except Exception as e:
        print(f'cannot process {complex_path}, error: {e}')

# %%
class PLIDataLoader(DataLoader):
    def __init__(self, data, **kwargs):
        super().__init__(data, collate_fn=data.collate_fn, **kwargs)

class GraphDataset(Dataset):
    """
    This class is used for generating graph objects using multi process
    """
    def __init__(self, data_dir, data_df, split_method, split, mmseqs_seq_clus_df, dis_threshold=5, graph_type='Graph_GIGN', num_process=8, create=False, seed=None):
        self.data_dir = data_dir
        self.data_df = data_df
        self.dis_threshold = dis_threshold
        self.graph_type = graph_type
        self.create = create
        self.graph_paths = None
        self.complex_ids = None
        self.num_process = num_process
        self.split_method = split_method
        self.split = split
        self.mmseqs_seq_clus_df = pd.read_table(mmseqs_seq_clus_df, names=['rep', 'seq']) 
        self.seed = seed
        self._pre_process()
        
       

    def _pre_process(self):
        data_dir = self.data_dir
        data_df = self.data_df
        graph_type = self.graph_type

        complex_path_list = []
        complex_id_list = []
        pKa_list = []
        graph_path_list = []
            
        
        split_frac=[0.7, 0.1, 0.2]
        
        ## the overlap position_dict and self.df

        if self.split_method == 'random':
            split_df = create_fold(data_df, self.seed, split_frac)
            self.train_graph_list, self.val_graph_list, self.test_graph_list = self.get_split(split_df)
        elif self.split_method == 'drug':
            split_df = create_fold_setting_cold(data_df, self.seed, split_frac, 'drug')
            self.train_graph_list, self.val_graph_list, self.test_graph_list = self.get_split(split_df)
        elif self.split_method == 'protein':
            split_df = create_fold_setting_cold(data_df, self.seed, split_frac, 'protein')
            self.train_graph_list, self.val_graph_list, self.test_graph_list = self.get_split(split_df)
        elif self.split_method == 'both':
            split_df = create_full_ood_set(data_df, self.seed, split_frac)
            self.train_graph_list, self.val_graph_list, self.test_graph_list = self.get_split(split_df)
        elif self.split_method == 'seqid':
            split_df = create_seq_identity_fold(data_df, self.mmseqs_seq_clus_df, self.seed, split_frac)
            self.train_graph_list, self.val_graph_list, self.test_graph_list = self.get_split(split_df)
        else:
            raise ValueError("Unknown split method: {}".format(self.split_method))
            

    def __getitem__(self, idx):
        if self.split == 'train':
            return torch.load(self.train_graph_list[idx])
        elif self.split == 'val':
            return torch.load(self.val_graph_list[idx])
        elif self.split == 'test':
            return torch.load(self.test_graph_list[idx])
        else:
            raise ValueError(f"Unknown split: {self.split}")
    
    def get_split(self, split_df):
        train_df, val_df, test_df = split_df['train'], split_df['valid'], split_df['test']
        train_graph_list = self.get_graph_list(train_df)
        val_graph_list = self.get_graph_list(val_df)
        test_graph_list = self.get_graph_list(test_df)
        return train_graph_list, val_graph_list, test_graph_list
   
    def get_graph_list(self, data_df):

        complex_path_list = []
        complex_id_list = []
        pKa_list = []
        graph_path_list = []
        dis_thresholds = repeat(self.dis_threshold, len(data_df))

        for i, row in data_df.iterrows():
            name = f'{row["protein"]}_{row["drug"]}'
            kd = row['y']
            complex_dir = os.path.join(self.data_dir, name)
            graph_path = os.path.join(complex_dir, f"{self.graph_type}-{name}_{self.dis_threshold}A.pyg")
            complex_path = os.path.join(complex_dir, f"{name}_{self.dis_threshold}A.rdkit")

            complex_path_list.append(complex_path)
            complex_id_list.append(name)
            pKa_list.append(kd)
            graph_path_list.append(graph_path)

        if self.create:
            print('Generate complex graph...')
            # multi-thread processing
            pool = multiprocessing.Pool(self.num_process)
            pool.starmap(mols2graphs,
                            zip(complex_path_list, pKa_list, graph_path_list, dis_thresholds))
            pool.close()
            pool.join()

        return [graph_path for graph_path in graph_path_list if os.path.exists(graph_path)]


    def collate_fn(self, batch):
        return Batch.from_data_list(batch)

    def __len__(self):
        if self.split == 'train':
            return len(self.train_graph_list)
        elif self.split == 'val':
            return len(self.val_graph_list)
        elif self.split == 'test':
            return len(self.test_graph_list)
        else:
            raise ValueError(f"Unknown split: {self.split}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Description of your script')
    parser.add_argument('--data_df', type=str, default='data/benchmark/davis_data.tsv', help='data of protein and ligand')
    parser.add_argument('--complex_path', type=str, default='data/benchmark/davis_complex_colabfold_diffdock', help='the path of the complexes')
    parser.add_argument('--mmseqs_seq_clus_df', type=str, default='data/benchmark/davis_cluster_id50_cluster.tsv', help='the path of mmseqs seq clus')
    args = parser.parse_args()


    data_root = args.complex_path
    data_df = args.data_df
    mmseqs_seq_clus_df= args.mmseqs_seq_clus_df
    data_df = pd.read_csv(data_df, sep='\t')
    
    drug_train = GraphDataset(data_root, data_df, split_method='drug', split='train', graph_type='Graph_GIGN', dis_threshold=5, mmseqs_seq_clus_df=mmseqs_seq_clus_df, create=True)
    drug_val = GraphDataset(data_root, data_df, split_method='drug', split='val', graph_type='Graph_GIGN', dis_threshold=5, mmseqs_seq_clus_df=mmseqs_seq_clus_df, create=True)
    drug_test = GraphDataset(data_root, data_df, split_method='drug', split='test', graph_type='Graph_GIGN', dis_threshold=5, mmseqs_seq_clus_df=mmseqs_seq_clus_df, create=True)
    
    print(f'split_method: drug')
    print(f"drug_train: {len(drug_train)}")
    print(f"drug_val: {len(drug_val)}")
    print(f"drug_test: {len(drug_test)}")

    protein_train = GraphDataset(data_root, data_df, split_method='protein', split='train', graph_type='Graph_GIGN', dis_threshold=5, mmseqs_seq_clus_df=mmseqs_seq_clus_df, create=True)
    protein_val = GraphDataset(data_root, data_df, split_method='protein', split='val', graph_type='Graph_GIGN', dis_threshold=5, mmseqs_seq_clus_df=mmseqs_seq_clus_df, create=True)
    protein_test = GraphDataset(data_root, data_df, split_method='protein', split='test', graph_type='Graph_GIGN', dis_threshold=5, mmseqs_seq_clus_df=mmseqs_seq_clus_df, create=True)
    
    print(f'split_method: protein')
    print(f"protein_train: {len(protein_train)}")
    print(f"protein_val: {len(protein_val)}")
    print(f"protein_test: {len(protein_test)}")

    both_train = GraphDataset(data_root, data_df, split_method='both', split='train', graph_type='Graph_GIGN', dis_threshold=5, mmseqs_seq_clus_df=mmseqs_seq_clus_df, create=True)
    both_val = GraphDataset(data_root, data_df, split_method='both', split='val', graph_type='Graph_GIGN', dis_threshold=5, mmseqs_seq_clus_df=mmseqs_seq_clus_df, create=True)
    both_test = GraphDataset(data_root, data_df, split_method='both', split='test', graph_type='Graph_GIGN', dis_threshold=5, mmseqs_seq_clus_df=mmseqs_seq_clus_df, create=True)
    
    print(f'split_method: both')
    print(f"both_train: {len(both_train)}")
    print(f"both_val: {len(both_val)}")
    print(f"both_test: {len(both_test)}")

    seqid_train = GraphDataset(data_root, data_df, split_method='seqid', split='train', graph_type='Graph_GIGN', dis_threshold=5, mmseqs_seq_clus_df=mmseqs_seq_clus_df, create=True)
    seqid_val = GraphDataset(data_root, data_df, split_method='seqid', split='val', graph_type='Graph_GIGN', dis_threshold=5, mmseqs_seq_clus_df=mmseqs_seq_clus_df, create=True)
    seqid_test = GraphDataset(data_root, data_df, split_method='seqid', split='test', graph_type='Graph_GIGN', dis_threshold=5, mmseqs_seq_clus_df=mmseqs_seq_clus_df, create=True)
    
    print(f'split_method: seqid')
    print(f"seqid_train: {len(drug_train)}")
    print(f"seqid_val: {len(drug_val)}")
    print(f"seqid_test: {len(drug_test)}")

    random_train = GraphDataset(data_root, data_df, split_method='random', split='train', graph_type='Graph_GIGN', dis_threshold=5, mmseqs_seq_clus_df=mmseqs_seq_clus_df, create=True)
    random_val = GraphDataset(data_root, data_df, split_method='random', split='val', graph_type='Graph_GIGN', dis_threshold=5, mmseqs_seq_clus_df=mmseqs_seq_clus_df, create=True)
    random_test = GraphDataset(data_root, data_df, split_method='random', split='test', graph_type='Graph_GIGN', dis_threshold=5, mmseqs_seq_clus_df=mmseqs_seq_clus_df, create=True)

    print(f'split_method: random')
    print(f"random_train: {len(random_train)}")
    print(f"random_val: {len(random_val)}")
    print(f"random_test: {len(random_test)}")
    
    

# %%
