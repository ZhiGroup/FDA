def add_train_args(parser):
    # Dataset parameters
    parser.add_argument('--task', help='Task name')
    parser.add_argument('--split_method', default='random',
        choices=['random', 'protein', 'drug', 'both', 'seqid'],
        help='Split method: random, protein, drug, or both')
    parser.add_argument('--seed', type=int, default=42,
        help='Random Seed')

    # Data representation parameters
    parser.add_argument('--contact_cutoff', type=float, default=8.,
        help='cutoff of C-alpha distance to define protein contact graph')
    parser.add_argument('--num_pos_emb', type=int, default=16,
        help='number of positional embeddings')
    parser.add_argument('--num_rbf', type=int, default=16,
        help='number of RBF kernels')

    # Protein model parameters
    parser.add_argument('--prot_gcn_dims', type=int, nargs='+', default=[128, 256, 256],
        help='protein GCN layers dimensions')
    parser.add_argument('--prot_fc_dims', type=int, nargs='+', default=[1024, 128],
        help='protein FC layers dimensions')

    # Drug model parameters
    parser.add_argument('--drug_gcn_dims', type=int, nargs='+', default=[128, 64],
        help='drug GVP hidden layers dimensions')
    parser.add_argument('--drug_fc_dims', type=int, nargs='+', default=[1024, 128],
        help='drug FC layers dimensions')

    # Top model parameters
    parser.add_argument('--mlp_dims', type=int, nargs='+', default=[1024, 512],
        help='top MLP layers dimensions')
    parser.add_argument('--mlp_dropout', type=float, default=0.25,
        help='dropout rate in top MLP')

    # uncertainty parameters
    parser.add_argument('--uncertainty', action='store_true',
        help='estimate uncertainty')
    parser.add_argument('--recalibrate', action='store_true',
        help='recalibrate uncertainty')

    # Training parameters
    parser.add_argument('--n_ensembles', type=int, default=1,
        help='number of ensembles')
    parser.add_argument('--batch_size', type=int, default=128,
        help='batch size')
    parser.add_argument('--n_epochs', type=int, default=500,
        help='number of epochs')
    parser.add_argument('--patience', action='store', type=int,
        help='patience for early stopping')
    parser.add_argument('--eval_freq', type=int, default=1,
        help='evaluation frequency')
    parser.add_argument('--test_freq', type=int,
        help='test frequency')
    parser.add_argument('--lr', type=float, default=0.0005,
        help='learning rate')
    parser.add_argument('--monitor_metric', default='pearson',
        help='validation metric to monitor for deciding best checkpoint')
    parser.add_argument('--parallel', action='store_true',
        help='run ensembles in parallel on multiple GPUs')

    # Save parameters
    parser.add_argument('--output_dir', action='store', default='../output', help='output folder')
    parser.add_argument('--save_log', action='store_true', default=False, help='save log file')
    parser.add_argument('--save_checkpoint', action='store_true', default=False, help='save checkpoint')
    parser.add_argument('--save_prediction', action='store_true', default=False, help='save prediction')
