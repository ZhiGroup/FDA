
# %%
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from utils import AverageMeter
from GIGN import GIGN
from dataset_GIGN_ablation import GraphDataset, PLIDataLoader
from config.config_dict import Config
from log.train_logger import TrainLogger
import numpy as np
from utils import *
from sklearn.metrics import mean_squared_error
from argparse import Namespace, ArgumentParser, FileType


# %%
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

# %%
if __name__ == '__main__':
    cfg = 'TrainConfig_GIGN_ablation'
    config = Config(cfg)
    args = config.get_config()
    graph_type = args.get("graph_type")
    save_model = args.get("save_model")
    batch_size = args.get("batch_size")
    epochs = args.get('epochs')
    repeats = args.get('repeat')
    early_stop_epoch = args.get("early_stop_epoch")
    logger = TrainLogger(args, cfg, create=True)
    

    parser = ArgumentParser()
    parser.add_argument('--scenario', type=str, required=True, choices=['crystal_crystal', 'crystal_diffdock', 'colabfold_diffdock'], default=None)
    parser.add_argument('--gpu', type=int, default=0)
    args_ = parser.parse_args()



    all_crystal_crystal_rmse = []
    all_crystal_diffdock_rmse = []
    all_colabfold_diffdock_rmse = []
    
    all_crystal_crystal_rp = []
    all_crystal_diffdock_rp = []
    all_colabfold_diffdock_rp = []

    data_root = '../../data/ablation_study'
    
    for repeat in range(repeats):
        args['repeat'] = repeat

        train_dir = os.path.join(data_root, args_.scenario, 'train')
        val_dir = os.path.join(data_root, args_.scenario, 'valid')
        
        train_df = pd.read_csv(os.path.join(data_root, 'train.csv'))
        val_df = pd.read_csv(os.path.join(data_root, 'valid.csv'))
        
        train_set = GraphDataset(train_dir, train_df, graph_type='Graph_GIGN', dis_threshold=5, create=False)
        valid_set = GraphDataset(val_dir, val_df, graph_type='Graph_GIGN', dis_threshold=5, create=False) 
  
        davis_ablation_crystal_crystal_dir = os.path.join(data_root, 'crystal_crystal', 'test')
        davis_ablation_crystal_diffdock_dir = os.path.join(data_root, 'crystal_diffdock', 'test')
        davis_ablation_colabfold_diffdock_dir = os.path.join(data_root, 'colabfold_diffdock', 'test')

        davis_ablation_df = pd.read_csv(os.path.join(data_root, 'test.csv'))

        davis_ablation_crystal_crystal = GraphDataset(davis_ablation_crystal_crystal_dir, davis_ablation_df, graph_type=graph_type, create=False)
        davis_ablation_crystal_diffdock = GraphDataset(davis_ablation_crystal_diffdock_dir, davis_ablation_df, graph_type=graph_type, create=False)
        davis_ablation_colabfold_diffdock = GraphDataset(davis_ablation_colabfold_diffdock_dir, davis_ablation_df, graph_type=graph_type, create=False)

        train_loader = PLIDataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
        valid_loader = PLIDataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=4)
        davis_ablation_crystal_crystal_loader = PLIDataLoader(davis_ablation_crystal_crystal, batch_size=batch_size, shuffle=False, num_workers=4)
        davis_ablation_crystal_diffdock_loader = PLIDataLoader(davis_ablation_crystal_diffdock, batch_size=batch_size, shuffle=False, num_workers=4)

        davis_ablation_colabfold_diffdock_loader = PLIDataLoader(davis_ablation_colabfold_diffdock, batch_size=batch_size, shuffle=False, num_workers=4)
        

        logger.info(f"this is the {repeat + 1}th repeat")
        logger.info(__file__)
        logger.info(f"train data: {len(train_set)}")
        logger.info(f"valid data: {len(valid_set)}")
        logger.info(f"davis_ablation_crystal_crystal data: {len(davis_ablation_crystal_crystal)}")
        logger.info(f"davis_ablation_crystal_diffdock data: {len(davis_ablation_crystal_diffdock)}")
        logger.info(f"davis_ablation_colabfold_diffdock data: {len(davis_ablation_colabfold_diffdock)}")

        device = torch.device(f'cuda:{str(args_.gpu)}')
        model = GIGN(35, 256).to(device)
        optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-6)
        criterion = nn.MSELoss()

        running_loss = AverageMeter()
        running_acc = AverageMeter()
        running_best_mse = BestMeter("min")
        best_model_list = []
        
        model.train()
        for epoch in range(epochs):
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

            # start validating
            valid_mse, valid_rmse, valid_pr = val(model, valid_loader, device)
            crystal_crystal_mse, crystal_crystal_rmse, crystal_crystal_pr = val(model, davis_ablation_crystal_crystal_loader, device)
            crystal_diffdock_mse, crystal_diffdock_rmse, crystal_diffdock_pr = val(model, davis_ablation_crystal_diffdock_loader, device)
            colabfold_diffdock_mse, colabfold_diffdock_rmse, colabfold_diffdock_pr = val(model, davis_ablation_colabfold_diffdock_loader, device)
            
            msg = "epoch-%d, train_loss-%.4f, train_rmse-%.4f, valid_mse-%.4f, valid_rmse-%.4f, valid_pr-%.4f, \
                   crystal_crystal_mse-%.4f, crystal_diffdock_mse-%.4f, colabfold_diffdock_mse-%.4f" \
                    % (epoch, epoch_loss, epoch_rmse, valid_mse, valid_rmse, valid_pr, crystal_crystal_mse, crystal_diffdock_mse, colabfold_diffdock_mse)
            logger.info(msg)

            if valid_rmse < running_best_mse.get_best():
                running_best_mse.update(valid_rmse)
                if save_model:
                    msg = "epoch-%d, train_loss-%.4f, train_rmse-%.4f, valid_mse-%.4f, valid_rmse-%.4f, valid_pr-%.4f" \
                    % (epoch, epoch_loss, epoch_rmse, valid_mse, valid_rmse, valid_pr)
                    model_path = os.path.join(logger.get_model_dir(), msg + '.pt')
                    best_model_list.append(model_path)
                    save_model_dict(model, logger.get_model_dir(), msg)
            else:
                count = running_best_mse.counter()
                if count > early_stop_epoch:
                    best_mse = running_best_mse.get_best()
                    msg = "best_rmse: %.4f" % best_mse
                    logger.info(f"early stop in epoch {epoch}")
                    logger.info(msg)
                    break_flag = True
                    break

        # final testing
        load_model_dict(model, best_model_list[-1])
        valid_mse, valid_rmse, valid_pr = val(model, valid_loader, device)
        crystal_crystal_mse, crystal_crystal_rmse, crystal_crystal_pr = val(model, davis_ablation_crystal_crystal_loader, device)
        crystal_diffdock_mse, crystal_diffdock_rmse, crystal_diffdock_pr = val(model, davis_ablation_crystal_diffdock_loader, device)
        colabfold_diffdock_mse, colabfold_diffdock_rmse, colabfold_diffdock_pr = val(model, davis_ablation_colabfold_diffdock_loader, device)
        
        all_crystal_crystal_rmse.append(crystal_crystal_rmse)
        all_crystal_diffdock_rmse.append(crystal_diffdock_rmse)
        all_colabfold_diffdock_rmse.append(colabfold_diffdock_rmse)

        
        all_crystal_crystal_rp.append(crystal_crystal_pr)
        all_crystal_diffdock_rp.append(crystal_diffdock_pr)
        all_colabfold_diffdock_rp.append(colabfold_diffdock_pr)
        

        msg = "valid_rmse-%.4f, valid_pr-%.4f, crystal_crystal_mse-%.4f, crystal_diffdock_mse-%.4f, colabfold_diffdock_mse-%.4f"\
               % (valid_rmse, valid_pr, crystal_crystal_mse, crystal_diffdock_mse, colabfold_diffdock_mse)

        logger.info(msg)
        
    logger.info(f"mean_crystal_crystal_rmse: {np.mean(all_crystal_crystal_rmse)}")
    logger.info(f"mean_crystal_diffdock_rmse: {np.mean(all_crystal_diffdock_rmse)}")
    logger.info(f"mean_colabfold_diffdock_rmse: {np.mean(all_colabfold_diffdock_rmse)}")
    logger.info(f"std_crystal_crystal_rmse: {np.std(all_crystal_crystal_rmse)}")
    logger.info(f"std_crystal_diffdock_rmse: {np.std(all_crystal_diffdock_rmse)}")
    logger.info(f"std_colabfold_diffdock_rmse: {np.std(all_colabfold_diffdock_rmse)}")
    
    logger.info(f"mean_crystal_crystal_rp: {np.mean(all_crystal_crystal_rp)}")
    logger.info(f"mean_crystal_diffdock_rp: {np.mean(all_crystal_diffdock_rp)}")
    logger.info(f"mean_colabfold_diffdock_rp: {np.mean(all_colabfold_diffdock_rp)}")
    logger.info(f"std_crystal_crystal_rp: {np.std(all_crystal_crystal_rp)}")
    logger.info(f"std_crystal_diffdock_rp: {np.std(all_crystal_diffdock_rp)}")
    logger.info(f"std_colabfold_diffdock_rp: {np.std(all_colabfold_diffdock_rp)}")
    