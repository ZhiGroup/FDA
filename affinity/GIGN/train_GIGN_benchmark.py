
# %%
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from utils import AverageMeter
from GIGN import GIGN
from dataset_GIGN_benchmark import GraphDataset, PLIDataLoader
from config.config_dict import Config
from log.train_logger import TrainLogger
import numpy as np
from utils import *
from sklearn.metrics import mean_squared_error
from argparse import Namespace, ArgumentParser, FileType
from joblib import Parallel, delayed, parallel_backend
from joblib.externals.loky.backend.context import get_context
import warnings
import pickle


warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")




# %%
def val_ensemble(models, dataloader, device):
    for model in models:
        model.eval()
    pred_list = []
    label_list = []
    for data in dataloader:
        data = data.to(device)
        label = data.y
        ensemble_pred = []
        for i in range(args_.ensemble_size):
            with torch.no_grad():
                model = models[i]
                pred = model(data)
                ensemble_pred.append(pred.detach().cpu().numpy())

        pred_list.append(np.mean(ensemble_pred, axis=0))
        label_list.append(label.detach().cpu().numpy())

    if not pred_list:
        return 0, 0, 0

    pred = np.concatenate(pred_list, axis=0)
    label = np.concatenate(label_list, axis=0)

    coff = np.corrcoef(pred, label)[0, 1]
    mse = mean_squared_error(label, pred)
    rmse = np.sqrt(mean_squared_error(label, pred))
    

    return mse, rmse, coff

def val(model, dataloader, device):
    model.eval()

    pred_list = []
    label_list = []
    for data in dataloader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data)
            label = data.y
            pred_list.append(pred.detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())
            
    pred = np.concatenate(pred_list, axis=0)
    label = np.concatenate(label_list, axis=0)

    coff = np.corrcoef(pred, label)[0, 1]
    mse = mean_squared_error(label, pred)
    rmse = np.sqrt(mean_squared_error(label, pred))
    

    model.train()

    return mse, rmse, coff


def train_one_model(job_name, model_idx, model, optimizer, criterion, running_loss, 
                    running_best_mse, best_model_list, seed, train_loader, valid_loader,
                    test_loader, device, epochs, early_stop_epoch, logger, save_model):
    """
    Train a single model for 'epochs' epochs and return the best RMSE or any stats you need.
    """
    break_flag = False
    for epoch in range(epochs):
        # Training loop
        model.train()
        for data in train_loader:
            data = data.to(device)
            pred = model(data)
            label = data.y

            loss = criterion(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss.update(loss.item(), label.size(0))

        epoch_loss = running_loss.get_average()
        epoch_rmse = np.sqrt(epoch_loss)
        running_loss.reset()

        # Validation
        valid_mse, valid_rmse, valid_rp = val(model, valid_loader, device)
        test_mse, test_rmse, test_rp = val(model, test_loader, device)
        msg = (
            f"[Model {model_idx}] epoch-{epoch}, train_loss-{epoch_loss:.4f}, "
            f"train_rmse-{epoch_rmse:.4f}, valid_mse-{valid_mse:.4f}, "
            f"valid_rmse-{valid_rmse:.4f}, valid_rp-{valid_rp:.4f}, "
            f"test_mse-{test_mse:.4f}, test_rmse-{test_rmse:.4f}, test_rp-{test_rp:.4f}"
        )
        print(msg)

        # Save best model
        if valid_rmse < running_best_mse.get_best():
            running_best_mse.update(valid_rmse)
            if save_model:
                # msg_save = (
                #     f"[Model {model_idx}] epoch-{epoch}, train_loss-{epoch_loss:.4f}, "
                #     f"train_rmse-{epoch_rmse:.4f}, valid_mse-{valid_mse:.4f}, "
                #     f"valid_rmse-{valid_rmse:.4f}, valid_rp-{valid_rp:.4f}"
                # )
                msg_save = f"model_{model_idx}_epoch_{epoch}"
                model_path = os.path.join('model', job_name, f'seed_{seed}', msg_save + '.pt')
                best_model_list.append(model_path)
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                torch.save(model.state_dict(), model_path)
                print("model has been saved to %s." % (model_path))
        else:
            count = running_best_mse.counter()
            if count > early_stop_epoch:
                best_mse = running_best_mse.get_best()
                logger.info(f"[Model {model_idx}] early stop in epoch {epoch}")
                logger.info(f"[Model {model_idx}] best_rmse: {best_mse:.4f}")
                break_flag = True
                break

        if break_flag:
            break

    # Return something useful, e.g., best RMSE, best models, etc.
    return {
        "model_idx": model_idx,
        "best_rmse": running_best_mse.get_best(),
        "model_paths": best_model_list
    }




# %%
if __name__ == '__main__':
    cfg = 'TrainConfig_GIGN_benchmark'
    config = Config(cfg)
    args = config.get_config()
    graph_type = args.get("graph_type")
    save_model = args.get("save_model")
    batch_size = args.get("batch_size")
    epochs = args.get('epochs')
    # repeats = args.get('repeat')
    early_stop_epoch = args.get("early_stop_epoch")
    # early_stop_epoch = 10
    logger = TrainLogger(args, cfg, create=True)
    
    parser = ArgumentParser()
    parser.add_argument('--split_method', type=str, default=None)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--data_df', type=str, default='../../data/benchmark/davis_data.tsv', help='data of protein and ligand')
    parser.add_argument('--complex_path', type=str, default='../../data/benchmark/davis_complex_colabfold_diffdock', help='the path of the complexes')
    parser.add_argument('--mmseqs_seq_clus_df', type=str, default='../../data/benchmark/davis_cluster_id50_cluster.tsv', help='the path of mmseqs seq clus')
    parser.add_argument('--ensemble_size', type=int, default=1)
    parser.add_argument('--job_name', type=str, default='benchmark_davis')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4], help='List of seeds for the repeats')
    parser.add_argument('--top_n', type=int, default=1, help='preprocess top n ligands')
    parser.add_argument('--protein_rank', nargs='+', type=int, default=None, help='List of protein rank')


    args_ = parser.parse_args()
    device = torch.device(f'cuda:{str(args_.gpu)}')
 
    seeds = args_.seeds
    job_name = args_.job_name
    top_n = args_.top_n
    data_root = args_.complex_path
    data_df = pd.read_csv(args_.data_df , sep='\t')
    protein_rank = args_.protein_rank

    for seed in seeds:

        if os.path.exists(f'model/{args_.job_name}/results_seed_{seed}.pkl'):
            continue
    
        train_set = GraphDataset(data_root, data_df, split_method=args_.split_method, split='train', graph_type='Graph_GIGN', top_n=top_n, protein_rank=protein_rank, dis_threshold=5, create=False, seed=seed, mmseqs_seq_clus_df=args_.mmseqs_seq_clus_df)
        val_set = GraphDataset(data_root, data_df, split_method=args_.split_method, split='val', graph_type='Graph_GIGN', top_n=top_n, protein_rank=protein_rank, dis_threshold=5, create=False, seed=seed, mmseqs_seq_clus_df=args_.mmseqs_seq_clus_df)
        test_set = GraphDataset(data_root, data_df, split_method=args_.split_method, split='test', graph_type='Graph_GIGN', top_n=1, protein_rank=None, dis_threshold=5, create=False, seed=seed, mmseqs_seq_clus_df=args_.mmseqs_seq_clus_df)
       
        train_loader = PLIDataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, multiprocessing_context=get_context('loky'))
        valid_loader = PLIDataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, multiprocessing_context=get_context('loky'))
        test_loader = PLIDataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, multiprocessing_context=get_context('loky'))
    
        

        logger.info(f"this is the seed {seed}")
        logger.info(__file__)
        logger.info(f"split method: {args_.split_method}")
        logger.info(f"train data: {len(train_set)}")
        logger.info(f"valid data: {len(val_set)}")
        logger.info(f"test data: {len(test_set)}")
    
        
        models = [GIGN(35, 256).to(device) for _ in range(args_.ensemble_size)]
        optimizers = [optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-6) for model in models]
        criterions = [nn.MSELoss() for _ in range(args_.ensemble_size)]

        running_losses = [AverageMeter() for _ in range(args_.ensemble_size)]
        running_best_mses = [BestMeter("min") for _ in range(args_.ensemble_size)]
        best_model_lists = [[] for _ in range(args_.ensemble_size)]
        
        pool_input = []
        for i in range(args_.ensemble_size):
            pool_input.append((job_name, i, models[i], 
                                optimizers[i], 
                                criterions[i], 
                                running_losses[i], 
                                running_best_mses[i], 
                                best_model_lists[i],
                                seed, 
                                train_loader, 
                                valid_loader, 
                                test_loader, 
                                device, 
                                epochs, 
                                early_stop_epoch, 
                                logger, 
                                save_model))
        with parallel_backend('loky', n_jobs=args_.ensemble_size):
            results = Parallel()(delayed(train_one_model)(*input_) for input_ in pool_input)


        with open(f'model/{job_name}/results_seed_{seed}.pkl', 'wb') as f:
            pickle.dump(results, f)


    all_test_mse = []
    all_test_rp = []

    for seed in seeds:

        train_set = GraphDataset(data_root, data_df, split_method=args_.split_method, split='train', graph_type='Graph_GIGN', top_n=top_n, dis_threshold=5, create=False, seed=seed, mmseqs_seq_clus_df=args_.mmseqs_seq_clus_df)
        val_set = GraphDataset(data_root, data_df, split_method=args_.split_method, split='val', graph_type='Graph_GIGN', top_n=top_n, dis_threshold=5, create=False, seed=seed, mmseqs_seq_clus_df=args_.mmseqs_seq_clus_df)
        test_set = GraphDataset(data_root, data_df, split_method=args_.split_method, split='test', graph_type='Graph_GIGN', top_n=1, dis_threshold=5, create=False, seed=seed, mmseqs_seq_clus_df=args_.mmseqs_seq_clus_df)


        train_loader = PLIDataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, multiprocessing_context=get_context('loky'))
        valid_loader = PLIDataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, multiprocessing_context=get_context('loky'))
        test_loader = PLIDataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, multiprocessing_context=get_context('loky'))
        test_wt_loader = PLIDataLoader(test_wt_set, batch_size=batch_size, shuffle=False, num_workers=4, multiprocessing_context=get_context('loky'))
        test_mutation_loader = PLIDataLoader(test_mutation_set, batch_size=batch_size, shuffle=False, num_workers=4, multiprocessing_context=get_context('loky'))


        models = [GIGN(35, 256).to(device) for _ in range(args_.ensemble_size)]
        results = read_pickle(f'model/{job_name}/results_seed_{seed}.pkl')
        for model, res in zip(models, results):
            model = models[res['model_idx']]
            load_model_dict(model, res['model_paths'][-1])

        valid_mse, valid_rmse, valid_rp = val_ensemble(models, valid_loader, device)
        test_mse, test_rmse, test_rp = val_ensemble(models, test_loader, device)

        msg = "valid_rmse-%.4f, valid_rp-%.4f, test_rmse-%.4f, test_rp-%.4f, test_wt_rmse-%.4f, test_wt_rp-%.4f, test_mutation_rmse-%.4f, test_mutation_rp-%.4f"\
            % (valid_rmse, valid_rp, test_rmse, test_rp, test_wt_rmse, test_wt_rp, test_mutation_rmse, test_mutation_rp)
        logger.info(msg)
    
        all_test_mse.append(test_mse)
        all_test_rp.append(test_rp)
    

    logger.info(f"mean test mse: {np.mean(all_test_mse)}")
    logger.info(f"std test mse: {np.std(all_test_mse)}") 
    logger.info(f"mean test rp: {np.mean(all_test_rp)}")
    logger.info(f"std test rp: {np.std(all_test_rp)}")
    
