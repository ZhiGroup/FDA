import rdkit.Chem
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
import torch_geometric
import torch_cluster

from kdbnet.constants import ATOM_VOCAB
from kdbnet.pdb_graph import _rbf, _normalize


def onehot_encoder(a=None, alphabet=None, default=None, drop_first=False):
    '''
    Parameters
    ----------
    a: array of numerical value of categorical feature classes.
    alphabet: valid values of feature classes.
    default: default class if out of alphabet.
    Returns
    -------
    A 2-D one-hot array with size |x| * |alphabet|
    '''
    # replace out-of-vocabulary classes
    alphabet_set = set(alphabet)
    a = [x if x in alphabet_set else default for x in a]

    # cast to category to force class not present
    a = pd.Categorical(a, categories=alphabet)

    onehot = pd.get_dummies(pd.Series(a), columns=alphabet, drop_first=drop_first)
    return onehot.values


def _build_atom_feature(mol):
    # dim: 44 + 7 + 7 + 7 + 1
    feature_alphabet = {
        # (alphabet, default value)
        'GetSymbol': (ATOM_VOCAB, 'unk'),
        'GetDegree': ([0, 1, 2, 3, 4, 5, 6], 6),
        'GetTotalNumHs': ([0, 1, 2, 3, 4, 5, 6], 6),
        'GetImplicitValence': ([0, 1, 2, 3, 4, 5, 6], 6),
        'GetIsAromatic': ([0, 1], 1)
    }

    atom_feature = None
    for attr in ['GetSymbol', 'GetDegree', 'GetTotalNumHs',
                'GetImplicitValence', 'GetIsAromatic']:
        feature = [getattr(atom, attr)() for atom in mol.GetAtoms()]
        feature = onehot_encoder(feature,
                    alphabet=feature_alphabet[attr][0],
                    default=feature_alphabet[attr][1],
                    drop_first=(attr in ['GetIsAromatic']) # binary-class feature
                )
        atom_feature = feature if atom_feature is None else np.concatenate((atom_feature, feature), axis=1)
    atom_feature = atom_feature.astype(np.float32)
    return atom_feature


def _build_edge_feature(coords, edge_index, D_max=4.5, num_rbf=16):
    E_vectors = coords[edge_index[0]] - coords[edge_index[1]]
    rbf = _rbf(E_vectors.norm(dim=-1), D_max=D_max, D_count=num_rbf)

    edge_s = rbf
    edge_v = _normalize(E_vectors).unsqueeze(-2)

    edge_s, edge_v = map(torch.nan_to_num, (edge_s, edge_v))

    return edge_s, edge_v


def sdf_to_graphs(data_list):
    """
    Parameters
    ----------
    data_list: dict, drug key -> sdf file path
    Returns
    -------
    graphs : dict
        A list of torch_geometric graphs. drug key -> graph
    """
    graphs = {}
    for key, sdf_path in tqdm(data_list.items(), desc='sdf'):
        graphs[key] = featurize_drug(sdf_path, name=key)
    return graphs


def featurize_drug(sdf_path, name=None, edge_cutoff=4.5, num_rbf=16):
    """
    Parameters
    ----------
    sdf_path: str
        Path to sdf file
    name: str
        Name of drug
    Returns
    -------
    graph: torch_geometric.data.Data
        A torch_geometric graph
    """
    mol = rdkit.Chem.MolFromMolFile(sdf_path)
    conf = mol.GetConformer()
    with torch.no_grad():
        coords = conf.GetPositions()
        coords = torch.as_tensor(coords, dtype=torch.float32)
        atom_feature = _build_atom_feature(mol)
        atom_feature = torch.as_tensor(atom_feature, dtype=torch.float32)
        edge_index = torch_cluster.radius_graph(coords, r=edge_cutoff)

    node_s = atom_feature
    node_v = coords.unsqueeze(1)
    # edge_v, edge_index = _build_edge_feature(mol)
    edge_s, edge_v = _build_edge_feature(
        coords, edge_index, D_max=edge_cutoff, num_rbf=num_rbf)

    data = torch_geometric.data.Data(
        x=coords, edge_index=edge_index, name=name,
        node_v=node_v, node_s=node_s, edge_v=edge_v, edge_s=edge_s)
    return data

