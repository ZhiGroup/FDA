
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
    cfg = 'TrainConfig_GIGN_benchmark'
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
    parser.add_argument('--split_method', type=str, default=None)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--data_df', type=str, default='../../data/benchmark/davis_data.tsv', help='data of protein and ligand')
    parser.add_argument('--complex_path', type=str, default='../../data/benchmark/davis_complex_colabfold_diffdock', help='the path of the complexes')
    parser.add_argument('--mmseqs_seq_clus_df', type=str, default='../../data/benchmark/davis_cluster_id50_cluster.tsv', help='the path of mmseqs seq clus')

    args_ = parser.parse_args()

    all_test_mse = []
    all_test_rp = []
 
    
    for repeat in range(repeats):
        args['repeat'] = repeat
        data_root = args_.complex_path
        data_df = pd.read_csv(args_.data_df , sep='\t')
    
        train_set = GraphDataset(data_root, data_df, split_method=args_.split_method, split='train', graph_type='Graph_GIGN', dis_threshold=5, create=False, seed=repeat, mmseqs_seq_clus_df=args_.mmseqs_seq_clus_df)
        val_set = GraphDataset(data_root, data_df, split_method=args_.split_method, split='val', graph_type='Graph_GIGN', dis_threshold=5, create=False, seed=repeat, mmseqs_seq_clus_df=args_.mmseqs_seq_clus_df)
        test_set = GraphDataset(data_root, data_df, split_method=args_.split_method, split='test', graph_type='Graph_GIGN', dis_threshold=5, create=False, seed=repeat, mmseqs_seq_clus_df=args_.mmseqs_seq_clus_df)

        train_loader = PLIDataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
        valid_loader = PLIDataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)
        test_loader = PLIDataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
        

        logger.info(f"this is the {repeat + 1}th repeat")
        logger.info(__file__)
        logger.info(f"split method: {args_.split_method}")
        logger.info(f"train data: {len(train_set)}")
        logger.info(f"valid data: {len(val_set)}")
        logger.info(f"test data: {len(test_set)}")
     

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
            test_mse, test_rmse, test_pr = val(model, test_loader, device)
            msg = "epoch-%d, train_loss-%.4f, train_rmse-%.4f, valid_mse-%.4f, valid_rmse-%.4f, valid_pr-%.4f, test_mse-%.4f, test_rmse-%.4f, test_pr-%.4f"\
                    % (epoch, epoch_loss, epoch_rmse, valid_mse, valid_rmse, valid_pr, test_mse, test_rmse, test_pr)
            logger.info(msg)

            if valid_rmse < running_best_mse.get_best():
                running_best_mse.update(valid_rmse)
                if save_model:
                    msg = "epoch-%d, train_loss-%.4f, train_rmse-%.4f, valid_mse-%.4f, valid_rmse-%.4f, valid_pr-%.4f"\
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
        test_mse, test_rmse, test_pr = val(model, test_loader, device)

        
        all_test_mse.append(test_mse)
        all_test_rp.append(test_pr)
        

        msg = "valid_rmse-%.4f, valid_pr-%.4f, test_rmse-%.4f, test_pr-%.4f"\
               % (valid_rmse, valid_pr, test_rmse, test_pr)
        logger.info(msg)
        
    logger.info(f"mean_test_mse: {np.mean(all_test_mse)}")
    logger.info(f"std_test_mse: {np.std(all_test_mse)}")
    
    logger.info(f"mean_test_rp: {np.mean(all_test_rp)}")
    logger.info(f"std_test_rp: {np.std(all_test_rp)}")
    
