import itertools
import math
import os
import pickle
import random
from argparse import Namespace, ArgumentParser, FileType
from functools import partial
import copy
from rdkit.Chem import RemoveHs

import numpy as np
import pandas as pd
import torch
import yaml
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader, DataListLoader
from tqdm import tqdm
import glob

from datasets.pdbbind_affinity_davis_colabfold import PDBBind
from utils.diffusion_utils import get_t_schedule
from utils.sampling import randomize_position, sampling
from utils.utils import get_model
from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl
from kdbnet.dta import create_fold, create_fold_setting_cold, create_full_ood_set, create_seq_identity_fold


class ListDataset(Dataset):
    def __init__(self, list):
        super().__init__()
        self.data_list = list

    def len(self) -> int:
        return len(self.data_list)

    def get(self, idx: int) -> Data:
        return self.data_list[idx]
    
class ListDataset_graph_split(Dataset):
    def __init__(self, list, calpha_root_path, allatoms_root_path=None):
        super().__init__()
        self.data_list = list
        self.calpha_root_path = calpha_root_path
        self.allatoms_root_path = allatoms_root_path
    def len(self) -> int:
        return len(self.data_list)

    def get(self, idx: int) -> Data:
        name = self.data_list[idx]
        calpha_heterograph_path = os.path.join(self.calpha_root_path, f'{name}.pkl')
        allatoms_heterograph_path = os.path.join(self.allatoms_root_path, f'{name}.pkl')
        return pickle.load(open(calpha_heterograph_path, 'rb')), pickle.load(open(allatoms_heterograph_path, 'rb'))


def get_cache_path(args, split, all_atoms):
    cache_path = args.cache_path
    if not args.no_torsion:
        cache_path += '_torsion'
    if all_atoms:
        cache_path += '_allatoms'
    
    cache_path = os.path.join(cache_path, f'limit{args.limit_complexes}_INDEX_{args.protein_dir_name}_{os.path.splitext(os.path.basename(args.split_train))[0]}_maxLigSize{args.max_lig_size}_H{int(not args.remove_hs)}_recRad{args.receptor_radius}_recMax{args.c_alpha_max_neighbors}'
                                       + ('' if not all_atoms else f'_atomRad{args.atom_radius}_atomMax{args.atom_max_neighbors}')
                                       + ('' if args.no_torsion or args.num_conformers == 1 else
                                           f'_confs{args.num_conformers}')
                              + ('' if args.esm_embeddings_path is None else f'_esmEmbeddings'))
    return cache_path

def get_args_and_cache_path(original_model_dir, split):
    with open(f'{original_model_dir}/model_parameters.yml') as f:
        model_args = Namespace(**yaml.full_load(f))
    return model_args, get_cache_path(model_args,split)

def get_model_args(original_model_dir):
    with open(f'{original_model_dir}/model_parameters.yml') as f:
        model_args = Namespace(**yaml.full_load(f))
    return model_args


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

class AffinityDataset(Dataset):
    def __init__(self, batch_size, cache_path, original_model_dir, confidence_model_dir, confidence_ckpt, split, device, limit_complexes,
                 inference_steps, samples_per_complex, no_random, ode, no_final_step_noise, all_atoms, split_method,
                 args, balance=False, use_original_model_cache=True, rmsd_classification_cutoff=2, no_parallel=True,
                 cache_ids_to_combine= None, cache_creation_id=None, heterographs_name=None, heterographs_split_size=None, heterographs_combine=None):

        super(AffinityDataset, self).__init__()

        self.device = device
        self.batch_size = batch_size
        self.inference_steps = inference_steps
        self.limit_complexes = limit_complexes
        self.all_atoms = all_atoms
        self.original_model_dir = original_model_dir
        self.balance = balance
        self.split_method = split_method 
        self.split = split
        # self.use_original_model_cache = use_original_model_cache
        self.rmsd_classification_cutoff = rmsd_classification_cutoff
        self.cache_ids_to_combine = cache_ids_to_combine
        self.cache_creation_id = cache_creation_id
        self.samples_per_complex = samples_per_complex
        self.no_random = no_random
        self.ode = ode
        self.confidence_model_dir = confidence_model_dir
        self.confidence_ckpt = confidence_ckpt
        self.no_final_step_noise = no_final_step_noise
        self.actual_steps = None
        self.heterographs_name = heterographs_name
        self.heterographs_split_size = heterographs_split_size
        self.heterographs_combine = heterographs_combine
        self.no_parallel = no_parallel 
        if args.confidence_model_dir is not None:
            with open(f'{args.confidence_model_dir}/model_parameters.yml') as f:
                self.confidence_model_args = Namespace(**yaml.full_load(f))
    
        self.original_model_args = get_model_args(original_model_dir)
        self.complex_graphs_allatom_cache = get_cache_path(args, 'train', all_atoms=True)
        self.complex_graphs_calpha_cache = get_cache_path(args, 'train', all_atoms=False)
        self.graph_split = args.graph_split
    
        # print('Using the cached complex graphs of the original model args' if self.use_original_model_cache else 'Not using the cached complex graphs of the original model args. Instead the complex graphs are used that are at the location given by the dataset parameters given to confidence_train.py')
        # print(self.complex_graphs_cache)
        # the original calpha complex graphs for scoring model
        if (not os.path.exists(os.path.join(self.complex_graphs_calpha_cache, "heterographs.pkl"))) and \
           (not glob.glob(os.path.join(self.complex_graphs_calpha_cache, "heterograph*.pkl"))):
            print(f'HAPPENING | Complex graphs path does not exist yet: {os.path.join(self.complex_graphs_calpha_cache, "heterographs.pkl")}. For that reason, we are now creating the dataset.')
            PDBBind(transform=None, root=args.data_dir, protein_dir_name=args.protein_dir_name, ligand_dir_name=args.ligand_dir_name,
                    limit_complexes=args.limit_complexes,
                    receptor_radius=args.receptor_radius,
                    cache_path=args.cache_path, split_path=args.split_train,
                    remove_hs=args.remove_hs, max_lig_size=args.max_lig_size,
                    c_alpha_max_neighbors=args.c_alpha_max_neighbors,
                    matching=not args.no_torsion, keep_original=True,
                    popsize=args.matching_popsize,
                    maxiter=args.matching_maxiter,
                    all_atoms=False,
                    atom_radius=args.atom_radius,
                    atom_max_neighbors=args.atom_max_neighbors,
                    esm_embeddings_path=args.esm_embeddings_path,
                    require_ligand=True,
                    graph_split=args.graph_split)
            
        print(f'HAPPENING | Loading calpha complex graphs from: {os.path.join(self.complex_graphs_calpha_cache, "heterographs.pkl")}')
        

        if self.graph_split:
            self.dataset_names_calpha = [os.path.splitext(name)[0] for name in os.listdir(self.complex_graphs_calpha_cache) if name.startswith('heterograph_')]
        else:
            with open(os.path.join(self.complex_graphs_calpha_cache, "heterographs.pkl"), 'rb') as f:
                self.complex_graphs_calpha = pickle.load(f)
            self.complex_graph_calpha_dict = {d.name: d for d in self.complex_graphs_calpha}
        
        # the original allatom complex graphs for confidence model
        if (not os.path.exists(os.path.join(self.complex_graphs_allatom_cache, "heterographs.pkl"))) and \
           (not glob.glob(os.path.join(self.complex_graphs_allatom_cache, "heterograph*.pkl"))):
            print(f'HAPPENING | Complex graphs path does not exist yet: {os.path.join(self.complex_graphs_allatom_cache, "heterographs.pkl")}. For that reason, we are now creating the dataset.')
            PDBBind(transform=None, root=args.data_dir, protein_dir_name=args.protein_dir_name, ligand_dir_name=args.ligand_dir_name,
                    limit_complexes=args.limit_complexes,
                    receptor_radius=args.receptor_radius,
                    cache_path=args.cache_path, split_path=args.split_train,
                    remove_hs=args.remove_hs, max_lig_size=args.max_lig_size,
                    c_alpha_max_neighbors=args.c_alpha_max_neighbors,
                    matching=not args.no_torsion, keep_original=True,
                    popsize=args.matching_popsize,
                    maxiter=args.matching_maxiter,
                    all_atoms=True,
                    atom_radius=args.atom_radius,
                    atom_max_neighbors=args.atom_max_neighbors,
                    esm_embeddings_path=args.esm_embeddings_path,
                    require_ligand=True,
                    graph_split=args.graph_split)

        print(f'HAPPENING | Loading allatom complex graphs from: {os.path.join(self.complex_graphs_allatom_cache, "heterographs.pkl")}')
       

        if self.graph_split:
            self.dataset_names_allatom = [os.path.splitext(name)[0] for name in os.listdir(self.complex_graphs_allatom_cache) if name.startswith('heterograph_')]
        else:
            with open(os.path.join(self.complex_graphs_allatom_cache, "heterographs.pkl"), 'rb') as f:
                self.complex_graphs_allatom = pickle.load(f)
            self.complex_graph_allatom_dict = {d.name: d for d in self.complex_graphs_allatom}
        
        
        if self.graph_split:
            assert len(self.dataset_names_calpha) == len(self.dataset_names_allatom)
            self.dataset_names = list(set(self.dataset_names_calpha) & set(self.dataset_names_allatom))
        else:
            assert len(self.complex_graph_calpha_dict) == len(self.complex_graph_allatom_dict)

        self.full_cache_path = os.path.join(cache_path, f'{args.protein_dir_name}_{os.path.splitext(os.path.basename(args.split_train))[0]}'
                                            f'_split_train_limit_{limit_complexes}')
        
        print(f'self.full_cache_path: {self.full_cache_path}')
        
        
        if not os.path.exists(os.path.join(self.full_cache_path, f"ligand_positions_rank1.pkl")):
            os.makedirs(self.full_cache_path, exist_ok=True)
            self.preprocessing()

        if self.cache_ids_to_combine is None:
            print(f'HAPPENING | Loading positions and rmsds from: {os.path.join(self.full_cache_path, "ligand_positions_rank1.pkl")}')
            with open(os.path.join(self.full_cache_path, "ligand_positions_rank1.pkl"), 'rb') as f:
                self.full_ligand_positions = pickle.load(f)
   
        else:
            all_rmsds_unsorted, all_full_ligand_positions_unsorted, all_names_unsorted = [], [], []
            for idx, cache_id in enumerate(self.cache_ids_to_combine):
                print(f'HAPPENING | Loading positions and rmsds from cache_id from the path: {os.path.join(self.full_cache_path, "ligand_positions_"+ str(cache_id)+ ".pkl")}')
                if not os.path.exists(os.path.join(self.full_cache_path, f"ligand_positions_id{cache_id}.pkl")): raise Exception(f'The generated ligand positions with cache_id do not exist: {cache_id}') # be careful with changing this error message since it is sometimes cought in a try catch
                with open(os.path.join(self.full_cache_path, f"ligand_positions_id{cache_id}.pkl"), 'rb') as f:
                    full_ligand_positions, rmsds = pickle.load(f)
                with open(os.path.join(self.full_cache_path, f"complex_names_in_same_order_id{cache_id}.pkl"), 'rb') as f:
                    names_unsorted = pickle.load(f)
                all_names_unsorted.append(names_unsorted)
                all_rmsds_unsorted.append(rmsds)
                all_full_ligand_positions_unsorted.append(full_ligand_positions)
            names_order = list(set(sum(all_names_unsorted, [])))
            all_rmsds, all_full_ligand_positions, all_names = [], [], []
            for idx, (rmsds_unsorted, full_ligand_positions_unsorted, names_unsorted) in enumerate(zip(all_rmsds_unsorted,all_full_ligand_positions_unsorted, all_names_unsorted)):
                name_to_pos_dict = {name: (rmsd, pos) for name, rmsd, pos in zip(names_unsorted, full_ligand_positions_unsorted, rmsds_unsorted) }
                intermediate_rmsds = [name_to_pos_dict[name][1] for name in names_order]
                all_rmsds.append((intermediate_rmsds))
                intermediate_pos = [name_to_pos_dict[name][0] for name in names_order]
                all_full_ligand_positions.append((intermediate_pos))
            self.full_ligand_positions, self.rmsds = [], []
            for positions_tuple in list(zip(*all_full_ligand_positions)):
                self.full_ligand_positions.append(np.concatenate(positions_tuple, axis=0))
            for positions_tuple in list(zip(*all_rmsds)):
                self.rmsds.append(np.concatenate(positions_tuple, axis=0))
            complex_names = names_order
            
        self.positions_dict = self.full_ligand_positions
     
        
      
        
    def len(self):
        return len(self.full_ligand_positions)
    

    def get(self, idx):
        return

    def preprocessing(self):
        t_to_sigma = partial(t_to_sigma_compl, args=self.original_model_args)

        model = get_model(self.original_model_args, self.device, t_to_sigma=t_to_sigma, no_parallel=True)
        state_dict = torch.load(f'{self.original_model_dir}/best_ema_inference_epoch_model.pt', map_location=torch.device('cpu'))
        model.load_state_dict(state_dict, strict=True)
        model = model.to(self.device)
        model.eval()
        
        if self.confidence_model_dir is not None:
            confidence_model = get_model(self.confidence_model_args, self.device, t_to_sigma=t_to_sigma, no_parallel=True, confidence_mode=True)
            state_dict = torch.load(f'{self.confidence_model_dir}/{self.confidence_ckpt}', map_location=torch.device('cpu'))
            confidence_model.load_state_dict(state_dict, strict=True)
            confidence_model = confidence_model.to(self.device)
            confidence_model.eval()
        else:
            confidence_model = None
            confidence_args = None
            confidence_model_args = None
        
        tr_schedule = get_t_schedule(inference_steps=self.inference_steps)
        rot_schedule = tr_schedule
        tor_schedule = tr_schedule
        print('common t schedule', tr_schedule)
        

        if self.graph_split:
            dataset = ListDataset_graph_split(self.dataset_names, calpha_root_path=self.complex_graphs_calpha_cache, allatoms_root_path=self.complex_graphs_allatom_cache)
        else:
            dataset = ListDataset(self.complex_graphs_calpha)

        loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

        if glob.glob(os.path.join(self.full_cache_path, '*_tmp.pkl')):

            with open(os.path.join(self.full_cache_path, 'ligand_positions_tmp.pkl'), 'rb') as f:
                full_ligand_positions = pickle.load(f)
            with open(os.path.join(self.full_cache_path, 'confidences_ligand_positions_tmp.pkl'), 'rb') as f:
                confidences_ligand_positions = pickle.load(f)
            with open(os.path.join(self.full_cache_path, 'ligand_positions_rank1_tmp.pkl'), 'rb') as f:
                full_ligand_positions_rank1 = pickle.load(f)
            with open(os.path.join(self.full_cache_path, 'confidence_rank1_tmp.pkl'), 'rb') as f:
                confidence_rank1 = pickle.load(f)
            with open(os.path.join(self.full_cache_path, 'complex_names_in_same_order_tmp.pkl'), 'rb') as f:
                names = pickle.load(f)

            print(f'HAPPENING | Loaded the temporary cache files. {len(names)} complexes have been processed so far. remaining: {len(dataset) - len(names)}')

        else:
            full_ligand_positions, confidences_ligand_positions, full_ligand_positions_rank1, confidence_rank1, names = {}, {}, {}, {}, set() 
        
        for idx, orig_complex_graph in tqdm(enumerate(loader)):

            if self.graph_split:
                orig_complex_graph, allatom_complex_graph = orig_complex_graph

            if orig_complex_graph.name[0] in names:
                continue

            data_list = [copy.deepcopy(orig_complex_graph) for _ in range(self.samples_per_complex)]
            randomize_position(data_list, self.original_model_args.no_torsion, False, self.original_model_args.tr_sigma_max)
            if confidence_model is not None and not (self.confidence_model_args.use_original_model_cache or self.confidence_model_args.transfer_weights):
                if self.graph_split:
                    confidence_data_list = [copy.deepcopy(allatom_complex_graph) for _ in range(self.samples_per_complex)]
                else: 
                    confidence_data_list = [copy.deepcopy(self.complex_graph_allatom_dict[orig_complex_graph.name[0]]) for _ in range(self.samples_per_complex)]
            else:
                confidence_data_list = None

            predictions_list = None
            failed_convergence_counter = 0
            while predictions_list is None:
                try:
                    predictions_list, confidences = sampling(data_list=data_list, model=model,
                                                             inference_steps=self.inference_steps if self.actual_steps is not None else self.inference_steps,
                                                             tr_schedule=tr_schedule, rot_schedule=rot_schedule, tor_schedule=tor_schedule,
                                                             device=self.device, t_to_sigma=t_to_sigma, model_args=self.original_model_args, no_random=self.no_random,
                                                             ode=self.ode, visualization_list=None, confidence_model=confidence_model,
                                                             confidence_data_list=confidence_data_list, confidence_model_args=self.confidence_model_args,
                                                             batch_size=self.batch_size, no_final_step_noise=self.no_final_step_noise)
                    
                    ligand_pos = np.asarray([complex_graph['ligand'].pos.cpu().numpy() + orig_complex_graph.original_center.cpu().numpy() for complex_graph in predictions_list])
                    if confidences is not None and isinstance(self.confidence_model_args.rmsd_classification_cutoff, list):
                        confidences = confidences[:,0]
                    if confidences is not None:
                        confidences = confidences.cpu().numpy()
                        re_order = np.argsort(confidences)[::-1]
                        confidences = confidences[re_order]
                        ligand_pos = ligand_pos[re_order]
                    # write_dir = f'{args.out_dir}/index{idx}_{data_list[0]["name"][0].replace("/","-")}'
                    # os.makedirs(write_dir, exist_ok=True)
                    # for rank, pos in enumerate(ligand_pos):
                        # mol_pred = copy.deepcopy(lig)
                        # if self.original_model_args.remove_hs: mol_pred = RemoveHs(mol_pred)
                        # if rank == 0:
                    full_ligand_positions_rank1[orig_complex_graph.name[0]] = ligand_pos[0]
                    confidence_rank1[orig_complex_graph.name[0]] = confidences[0]
                except Exception as e:
                    if 'failed to converge' in str(e):
                        failed_convergence_counter += 1
                        if failed_convergence_counter > 5:
                            print('| WARNING: SVD failed to converge 5 times - skipping the complex')
                            break
                        print('| WARNING: SVD failed to converge - trying again with a new sample')
                    else:
                        raise e
            if failed_convergence_counter > 5: predictions_list = data_list
            if self.original_model_args.no_torsion:
                orig_complex_graph['ligand'].orig_pos = (orig_complex_graph['ligand'].pos.cpu().numpy() + orig_complex_graph.original_center.cpu().numpy())

            if failed_convergence_counter > 5:
                pass
            else:
                names.add(orig_complex_graph.name[0])
                confidences_ligand_positions[orig_complex_graph.name[0]] = confidences
                full_ligand_positions[orig_complex_graph.name[0]] = ligand_pos
            assert(len(orig_complex_graph.name) == 1) # I just put this assert here because of the above line where I assumed that the list is always only lenght 1. Just in case it isn't maybe check what the names in there are.

            if idx % 100 == 0:

                with open(os.path.join(self.full_cache_path, 'ligand_positions_tmp.pkl'), 'wb') as f:
                    pickle.dump((full_ligand_positions), f)
                with open(os.path.join(self.full_cache_path, 'confidences_ligand_positions_tmp.pkl'), 'wb') as f:
                    pickle.dump((confidences_ligand_positions), f)
                with open(os.path.join(self.full_cache_path, 'ligand_positions_rank1_tmp.pkl'), 'wb') as f:
                    pickle.dump((full_ligand_positions_rank1), f)  
                with open(os.path.join(self.full_cache_path, 'confidence_rank1_tmp.pkl'), 'wb') as f:
                    pickle.dump((confidence_rank1), f)
                with open(os.path.join(self.full_cache_path, 'complex_names_in_same_order_tmp.pkl'), 'wb') as f:
                    pickle.dump((names), f)
        
        with open(os.path.join(self.full_cache_path, f"ligand_positions_rank1{'' if self.cache_creation_id is None else '_id' + str(self.cache_creation_id)}{'' if self.heterographs_name is None else '_'+ str(self.heterographs_name)}.pkl"), 'wb') as f:
            pickle.dump((full_ligand_positions_rank1), f)
        with open(os.path.join(self.full_cache_path, f"ligand_positions{'' if self.cache_creation_id is None else '_id' + str(self.cache_creation_id)}{'' if self.heterographs_name is None else '_'+ str(self.heterographs_name)}.pkl"), 'wb') as f:
            pickle.dump((full_ligand_positions), f)
            
        with open(os.path.join(self.full_cache_path, f"confidence_rank1{'' if self.cache_creation_id is None else '_id' + str(self.cache_creation_id)}{'' if self.heterographs_name is None else '_'+ str(self.heterographs_name)}.pkl"), 'wb') as f:
            pickle.dump((confidence_rank1), f)
        with open(os.path.join(self.full_cache_path, f"confidences_ligand_positions{'' if self.cache_creation_id is None else '_id' + str(self.cache_creation_id)}{'' if self.heterographs_name is None else '_'+ str(self.heterographs_name)}.pkl"), 'wb') as f:
            pickle.dump((confidences_ligand_positions), f)
        
        with open(os.path.join(self.full_cache_path, f"complex_names_in_same_order{'' if self.cache_creation_id is None else '_id' + str(self.cache_creation_id)}{'' if self.heterographs_name is None else '_'+ str(self.heterographs_name)}.pkl"), 'wb') as f:
            pickle.dump((names), f)



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=FileType(mode='r'), default=None)
    parser.add_argument('--original_model_dir', type=str, default='workdir/paper_score_model', help='Path to folder with trained model and hyperparameters')
    parser.add_argument('--restart_dir', type=str, default=None, help='')
    parser.add_argument('--use_original_model_cache', action='store_true', default=False, help='If this is true, the same dataset as in the original model will be used. Otherwise, the dataset parameters are used.')
    parser.add_argument('--data_dir', type=str, default='../../data/benchmark', help='Folder containing original structures')
    parser.add_argument('--protein_dir_name', type=str, default='davis_colabfold_protein', help='Folder containing original structures')
    parser.add_argument('--ligand_dir_name', type=str, default='davis_ligand', help='Folder containing original structures')
    parser.add_argument('--ckpt', type=str, default='best_model.pt', help='Checkpoint to use inside the folder')
    parser.add_argument('--model_save_frequency', type=int, default=0, help='Frequency with which to save the last model. If 0, then only the early stopping criterion best model is saved and overwritten.')
    parser.add_argument('--best_model_save_frequency', type=int, default=0, help='Frequency with which to save the best model. If 0, then only the early stopping criterion best model is saved and overwritten.')
    parser.add_argument('--run_name', type=str, default='test_confidence', help='')
    parser.add_argument('--project', type=str, default='diffdock_confidence', help='')
    parser.add_argument('--split_train', type=str, default='../../data/benchmark/davis_data.tsv', help='Path of file defining the split')
    parser.add_argument('--split_val', type=str, default='data/davis_remove_special_protein_ablation_val.csv', help='Path of file defining the split')
    parser.add_argument('--split_test', type=str, default='data/davis_remove_special_protein_ablation_test.csv', help='Path of file defining the split')
    parser.add_argument('--confidence_model_dir', type=str, default='workdir/paper_confidence_model', help='Path to folder with trained confidence model and hyperparameters')
    parser.add_argument('--confidence_ckpt', type=str, default='best_model_epoch75.pt', help='Checkpoint to use for the confidence model')
    parser.add_argument('--no_random', action='store_true', default=False, help='Use no randomness in reverse diffusion')
    parser.add_argument('--ode', action='store_true', default=False, help='Use ODE formulation for inference')
    parser.add_argument('--no_final_step_noise', action='store_true', default=False, help='Use no noise in the final step of the reverse diffusion')
    parser.add_argument('--gpu_num', type=int, default=0, help='assign the number of gpu')

    # parallel
    parser.add_argument('--heterographs_name', type=str, default=None, help='the name of splitting heterographs')
    parser.add_argument('--heterographs_split_size', type=int, default=None, help='the name of splitting heterographs')
    parser.add_argument('--heterographs_combine', action='store_true', default=False, help='combine id')

    # Inference parameters for creating the positions and rmsds that the confidence predictor will be trained on.
    parser.add_argument('--cache_path', type=str, default='data/cacheNew', help='Folder from where to load/restore cached dataset')
    parser.add_argument('--cache_ids_to_combine', nargs='+', type=str, default=None, help='RMSD value below which a prediction is considered a postitive. This can also be multiple cutoffs.')
    parser.add_argument('--cache_creation_id', type=int, default=None, help='number of times that inference is run on the full dataset before concatenating it and coming up with the full confidence dataset')
    parser.add_argument('--wandb', action='store_true', default=False, help='')
    parser.add_argument('--inference_steps', type=int, default=2, help='Number of denoising steps')
    parser.add_argument('--samples_per_complex', type=int, default=3, help='')
    parser.add_argument('--balance', action='store_true', default=False, help='If this is true than we do not force the samples seen during training to be the same amount of negatives as positives')
    parser.add_argument('--affinity_prediction', action='store_true', default=True, help='')
    parser.add_argument('--rmsd_classification_cutoff', nargs='+', type=float, default=2, help='RMSD value below which a prediction is considered a postitive. This can also be multiple cutoffs.')

    parser.add_argument('--log_dir', type=str, default='workdir', help='')
    parser.add_argument('--main_metric', type=str, default='confidence_loss', help='Metric to track for early stopping. Mostly [loss, accuracy, ROC AUC]')
    parser.add_argument('--main_metric_goal', type=str, default='max', help='Can be [min, max]')
    parser.add_argument('--transfer_weights', action='store_true', default=False, help='')
    parser.add_argument('--batch_size', type=int, default=8, help='')
    parser.add_argument('--lr', type=float, default=1e-3, help='')
    parser.add_argument('--w_decay', type=float, default=0.0, help='')
    parser.add_argument('--scheduler', type=str, default='plateau', help='')
    parser.add_argument('--scheduler_patience', type=int, default=20, help='')
    parser.add_argument('--n_epochs', type=int, default=5, help='')

    # Dataset
    parser.add_argument('--limit_complexes', type=int, default=0, help='')
    parser.add_argument('--all_atoms', action='store_true', default=True, help='')
    parser.add_argument('--multiplicity', type=int, default=1, help='')
    parser.add_argument('--chain_cutoff', type=float, default=10, help='')
    parser.add_argument('--receptor_radius', type=float, default=30, help='')
    parser.add_argument('--c_alpha_max_neighbors', type=int, default=10, help='')
    parser.add_argument('--atom_radius', type=float, default=5, help='')
    parser.add_argument('--atom_max_neighbors', type=int, default=8, help='')
    parser.add_argument('--matching_popsize', type=int, default=20, help='')
    parser.add_argument('--matching_maxiter', type=int, default=20, help='')
    parser.add_argument('--max_lig_size', type=int, default=100, help='Maximum number of heavy atoms')
    parser.add_argument('--remove_hs', action='store_true', default=False, help='remove Hs')
    parser.add_argument('--num_conformers', type=int, default=1, help='')
    parser.add_argument('--esm_embeddings_path', type=str, default='data/esm2_3billion_embeddings_davis_colabfold.pt',help='If this is set then the LM embeddings at that path will be used for the receptor features')
    parser.add_argument('--no_torsion', action='store_true', default=False, help='')
    parser.add_argument('--graph_split', action='store_true', default=False, help='save individual graphs and load them individually')

    # Model
    parser.add_argument('--num_conv_layers', type=int, default=2, help='Number of interaction layers')
    parser.add_argument('--max_radius', type=float, default=5.0, help='Radius cutoff for geometric graph')
    parser.add_argument('--scale_by_sigma', action='store_true', default=True, help='Whether to normalise the score')
    parser.add_argument('--ns', type=int, default=16, help='Number of hidden features per node of order 0')
    parser.add_argument('--nv', type=int, default=4, help='Number of hidden features per node of order >0')
    parser.add_argument('--distance_embed_dim', type=int, default=32, help='')
    parser.add_argument('--cross_distance_embed_dim', type=int, default=32, help='')
    parser.add_argument('--no_batch_norm', action='store_true', default=False, help='If set, it removes the batch norm')
    parser.add_argument('--use_second_order_repr', action='store_true', default=False, help='Whether to use only up to first order representations or also second')
    parser.add_argument('--cross_max_distance', type=float, default=80, help='')
    parser.add_argument('--dynamic_max_cross', action='store_true', default=False, help='')
    parser.add_argument('--dropout', type=float, default=0.0, help='MLP dropout')
    parser.add_argument('--embedding_type', type=str, default="sinusoidal", help='')
    parser.add_argument('--sigma_embed_dim', type=int, default=32, help='')
    parser.add_argument('--embedding_scale', type=int, default=10000, help='')
    parser.add_argument('--confidence_no_batchnorm', action='store_true', default=False, help='')
    parser.add_argument('--confidence_dropout', type=float, default=0.0, help='MLP dropout in confidence readout')
    
    args = parser.parse_args()
    device = torch.device(f'cuda:{args.gpu_num}' if torch.cuda.is_available() else 'cpu')
    common_args = {'batch_size': args.samples_per_complex, 'cache_path': args.cache_path, 'original_model_dir': args.original_model_dir,
                   'confidence_model_dir': args.confidence_model_dir, 'confidence_ckpt': args.confidence_ckpt,
                   'limit_complexes': args.limit_complexes, 'inference_steps': args.inference_steps,
                   'samples_per_complex': args.samples_per_complex, 
                   'all_atoms': args.all_atoms,
                   'no_random': args.no_random, 'ode': args.ode, 'no_final_step_noise': args.no_final_step_noise,
                   'rmsd_classification_cutoff': args.rmsd_classification_cutoff,
                   'use_original_model_cache': args.use_original_model_cache, 'cache_creation_id': args.cache_creation_id, 
                   "cache_ids_to_combine": args.cache_ids_to_combine, "heterographs_name": args.heterographs_name,
                   "heterographs_split_size": args.heterographs_split_size, "heterographs_combine": args.heterographs_combine}
    
    loader_class = DataListLoader if torch.cuda.is_available() else DataLoader
    train_dataset = AffinityDataset(split="train", split_method='drug', device=device, args=args, **common_args)
