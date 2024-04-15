import copy
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import uncertainty_toolbox as uct

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch_geometric
torch.set_num_threads(1)

from kdbnet.dta import KIBA, DAVIS
from kdbnet.model import DTAModel
from kdbnet.utils import(
    Logger,
    Saver,
    EarlyStopping
)
from kdbnet.metrics import evaluation_metrics


def _parallel_train_per_epoch(
    kwargs=None, test_loader=None,
    n_epochs=None, eval_freq=None, test_freq=None,
    monitoring_score='pearson',
    loss_fn=None, logger=None,
    test_after_train=True,
):
    midx = kwargs['midx']
    model = kwargs['model']
    optimizer = kwargs['optimizer']
    train_loader = kwargs['train_loader']
    valid_loader = kwargs['valid_loader']
    device = kwargs['device']
    stopper = kwargs['stopper']
    best_model_state_dict = kwargs['best_model_state_dict']
    if stopper.early_stop:
        return kwargs

    model.train()
    for epoch in range(1, n_epochs + 1):
        total_loss = 0
        for step, batch in enumerate(train_loader, start=1):
            xd = batch['drug'].to(device)
            xp = batch['protein'].to(device)
            y = batch['y'].to(device)
            optimizer.zero_grad()
            yh = model(xd, xp)
            loss = loss_fn(yh, y.view(-1, 1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / step
        if epoch % eval_freq == 0:
            val_results = _parallel_test(
                {'model': model, 'midx': midx, 'test_loader': valid_loader, 'device': device},
                loss_fn=loss_fn, logger=logger
            )
            is_best = stopper.update(val_results['metrics'][monitoring_score])
            if is_best:
                best_model_state_dict = copy.deepcopy(model.state_dict())
            logger.info(f"M-{midx} E-{epoch} | Train Loss: {train_loss:.4f} | Valid Loss: {val_results['loss']:.4f} | "\
                + ' | '.join([f'{k}: {v:.4f}' for k, v in val_results['metrics'].items()])
                + f" | best {monitoring_score}: {stopper.best_score:.4f}"
                )
        if test_freq is not None and epoch % test_freq == 0:
            test_results = _parallel_test(
                {'midx': midx, 'model': model, 'test_loader': test_loader, 'device': device},
                loss_fn=loss_fn, logger=logger
            )
            logger.info(f"M-{midx} E-{epoch} | Test Loss: {test_results['loss']:.4f} | "\
                + ' | '.join([f'{k}: {v:.4f}' for k, v in test_results['metrics'].items()])
                )

        if stopper.early_stop:
            logger.info('Eearly stop at epoch {}'.format(epoch))

    if best_model_state_dict is not None:
        model.load_state_dict(best_model_state_dict)
    if test_after_train:
        test_results = _parallel_test(
            {'midx': midx, 'model': model, 'test_loader': test_loader, 'device': device},
            loss_fn=loss_fn,
            test_tag=f"Model {midx}", print_log=True, logger=logger
        )
    rets = dict(midx = midx, model = model)
    return rets


def _parallel_test(
    kwargs=None, loss_fn=None, 
    test_tag=None, print_log=False, logger=None,
):
    midx = kwargs['midx']
    model = kwargs['model']
    test_loader = kwargs['test_loader']
    device = kwargs['device']
    model.eval()
    yt, yp, total_loss = torch.Tensor(), torch.Tensor(), 0
    with torch.no_grad():
        for step, batch in enumerate(test_loader, start=1):
            xd = batch['drug'].to(device)
            xp = batch['protein'].to(device)
            y = batch['y'].to(device)
            yh = model(xd, xp)
            loss = loss_fn(yh, y.view(-1, 1))
            total_loss += loss.item()
            yp = torch.cat([yp, yh.detach().cpu()], dim=0)
            yt = torch.cat([yt, y.detach().cpu()], dim=0)
    yt = yt.numpy()
    yp = yp.view(-1).numpy()
    results = {
        'midx': midx,
        'y_true': yt,
        'y_pred': yp,
        'loss': total_loss / step,
    }
    eval_metrics = evaluation_metrics(
        yt, yp,
        eval_metrics=['mse', 'spearman', 'pearson']
    )
    results['metrics'] = eval_metrics
    if print_log:
        logger.info(f"{test_tag} | Test Loss: {results['loss']:.4f} | "\
            + ' | '.join([f'{k}: {v:.4f}' for k, v in results['metrics'].items()]))
    return results


def _unpack_evidential_output(output):
    mu, v, alpha, beta = torch.split(output, output.shape[1]//4, dim=1)
    inverse_evidence = 1. / ((alpha - 1) * v)
    var = beta * inverse_evidence
    return mu, var, inverse_evidence


class DTAExperiment(object):
    def __init__(self,
        task=None,
        split_method='protein',
        split_frac=[0.7, 0.1, 0.2],
        prot_gcn_dims=[128, 128, 128], prot_gcn_bn=False,
        prot_fc_dims=[1024, 128],
        drug_in_dim=66, drug_fc_dims=[1024, 128], drug_gcn_dims=[128, 64],
        mlp_dims=[1024, 512], mlp_dropout=0.25,
        num_pos_emb=16, num_rbf=16,
        contact_cutoff=8.,
        n_ensembles=1, n_epochs=500, batch_size=256,
        lr=0.001,        
        seed=42, onthefly=False,
        uncertainty=False, parallel=False,
        output_dir='../output', save_log=False
    ):
        self.saver = Saver(output_dir)
        self.logger = Logger(logfile=self.saver.save_dir/'exp.log' if save_log else None)

        self.uncertainty = uncertainty
        self.parallel = parallel
        self.n_ensembles = n_ensembles
        if self.uncertainty and self.n_ensembles < 2:
            raise ValueError('n_ensembles must be greater than 1 when uncertainty is True')            
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        dataset_klass = {
            'kiba': KIBA,
            'davis': DAVIS,
        }[task]

        self.dataset = dataset_klass(
            split_method=split_method,
            split_frac=split_frac,
            seed=seed,
            onthefly=onthefly,
            num_pos_emb=num_pos_emb,
            num_rbf=num_rbf,
            contact_cutoff=contact_cutoff,
        )
        self._task_data_df_split = None
        self._task_loader = None

        n_gpus = torch.cuda.device_count()
        if self.parallel and n_gpus < self.n_ensembles:
            self.logger.warning(f"Visible GPUs ({n_gpus}) is fewer than "
            f"number of models ({self.n_ensembles}). Some models will be run on the same GPU"
            )
        self.devices = [torch.device(f'cuda:{i % n_gpus}')
            for i in range(self.n_ensembles)]
        self.model_config = dict(
            prot_emb_dim=1280,
            prot_gcn_dims=prot_gcn_dims,            
            prot_fc_dims=prot_fc_dims,
            drug_node_in_dim=[66, 1], 
            drug_node_h_dims=drug_gcn_dims,
            drug_fc_dims=drug_fc_dims,            
            mlp_dims=mlp_dims, mlp_dropout=mlp_dropout)
        self.build_model()
        self.criterion = F.mse_loss

        self.split_method = split_method
        self.split_frac = split_frac

        self.logger.info(self.models[0])
        self.logger.info(self.optimizers[0])

    def build_model(self):
        self.models = [DTAModel(**self.model_config).to(self.devices[i])
                        for i in range(self.n_ensembles)]
        self.optimizers = [optim.Adam(model.parameters(), lr=self.lr) for model in self.models]

    def _get_data_loader(self, dataset, shuffle=False):
        return torch_geometric.loader.DataLoader(
                    dataset=dataset,
                    batch_size=self.batch_size,
                    shuffle=shuffle,
                    pin_memory=False,
                    num_workers=0,
                )

    @property
    def task_data_df_split(self):
        if self._task_data_df_split is None:
            (data, df) = self.dataset.get_split(return_df=True)
            self._task_data_df_split = (data, df)
        return self._task_data_df_split

    @property
    def task_data(self):
        return self.task_data_df_split[0]

    @property
    def task_df(self):
        return self.task_data_df_split[1]

    @property
    def task_loader(self):
        if self._task_loader is None:
            _loader = {
                s: self._get_data_loader(
                    self.task_data[s], shuffle=(s == 'train'))
                for s in self.task_data
            }
            self._task_loader = _loader
        return self._task_loader

    def recalibrate_std(self, df, recalib_df):
        y_mean = recalib_df['y_pred'].values
        y_std = recalib_df['y_std'].values
        y_true = recalib_df['y_true'].values
        std_ratio = uct.recalibration.optimize_recalibration_ratio(
            y_mean, y_std, y_true, criterion="miscal")
        df['y_std_recalib'] = df['y_std'] * std_ratio
        return df

    def _format_predict_df(self, results,
            test_df=None, esb_yp=None, recalib_df=None):
        """
        results: dict with keys y_pred, y_true, y_var
        """
        df = self.task_df['test'].copy() if test_df is None else test_df.copy()
        assert np.allclose(results['y_true'], df['y'].values)
        df = df.rename(columns={'y': 'y_true'})
        df['y_pred'] = results['y_pred']
        if esb_yp is not None:
            if self.uncertainty:
                df['y_std'] = np.std(esb_yp, axis=0)
                if recalib_df is not None:
                    df = self.recalibrate_std(df, recalib_df)
            for i in range(self.n_ensembles):
                df[f'y_pred_{i + 1}'] = esb_yp[i]
        return df

    def train(self, n_epochs=None, patience=None,
                eval_freq=1, test_freq=None,
                monitoring_score='pearson',
                train_data=None, valid_data=None,                
                rebuild_model=False,
                test_after_train=False):
        n_epochs = n_epochs or self.n_epochs
        if rebuild_model:
            self.build_model()
        tl, vl = self.task_loader['train'], self.task_loader['valid']
        rets_list = []
        for i in range(self.n_ensembles):
            stp = EarlyStopping(eval_freq=eval_freq, patience=patience,
                                    higher_better=(monitoring_score != 'mse'))
            rets = dict(
                midx = i + 1,
                model = self.models[i],
                optimizer = self.optimizers[i],
                device = self.devices[i],
                train_loader = tl,
                valid_loader = vl,
                stopper = stp,
                best_model_state_dict = None,
            )
            rets_list.append(rets)

        rets_list = Parallel(n_jobs=(self.n_ensembles if self.parallel else 1), prefer="threads")(
            delayed(_parallel_train_per_epoch)(
                kwargs=rets_list[i],
                test_loader=self.task_loader['test'],
                n_epochs=n_epochs, eval_freq=eval_freq, test_freq=test_freq,
                monitoring_score=monitoring_score,
                loss_fn=self.criterion, logger=self.logger,
                test_after_train=test_after_train,
            ) for i in range(self.n_ensembles))

        for i, rets in enumerate(rets_list):
            self.models[rets['midx'] - 1] = rets['model']


    def test(self, test_model=None, test_loader=None,
                test_data=None, test_df=None,
                recalib_df=None,
                save_prediction=False, save_df_name='prediction.tsv',
                test_tag=None, print_log=False):
        test_models = self.models if test_model is None else [test_model]
        if test_data is not None:
            assert test_df is not None, 'test_df must be provided if test_data used'
            test_loader = self._get_data_loader(test_data)
        elif test_loader is not None:
            assert test_df is not None, 'test_df must be provided if test_loader used'
        else:
            test_loader = self.task_loader['test']
        rets_list = []
        for i, model in enumerate(test_models):
            rets = _parallel_test(
                kwargs={
                    'midx': i + 1,
                    'model': model,
                    'test_loader': test_loader,
                    'device': self.devices[i],
                },
                loss_fn=self.criterion,
                test_tag=f"Model {i+1}", print_log=True, logger=self.logger
            )
            rets_list.append(rets)


        esb_yp, esb_loss = None, 0
        for rets in rets_list:
            esb_yp = rets['y_pred'].reshape(1, -1) if esb_yp is None else\
                np.vstack((esb_yp, rets['y_pred'].reshape(1, -1)))
            esb_loss += rets['loss']

        y_true = rets['y_true']
        y_pred = np.mean(esb_yp, axis=0)
        esb_loss /= len(test_models)
        results = {
            'y_true': y_true,
            'y_pred': y_pred,
            'loss': esb_loss,
        }

        eval_metrics = evaluation_metrics(
            y_true, y_pred,
            eval_metrics=['mse', 'spearman', 'pearson']
        )
        results['metrics'] = eval_metrics
        results['df'] = self._format_predict_df(results,
            test_df=test_df, esb_yp=esb_yp, recalib_df=recalib_df)
        if save_prediction:
            self.saver.save_df(results['df'], save_df_name, float_format='%g')
        if print_log:
            self.logger.info(f"{test_tag} | Test Loss: {results['loss']:.4f} | "\
                + ' | '.join([f'{k}: {v:.4f}' for k, v in results['metrics'].items()]))
        return results
 