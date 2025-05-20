import argparse

from models.LightGCNManager import LightGCNManager


def parse_args():
    parser = argparse.ArgumentParser(description="LightGCN Training Arguments")

    parser.add_argument('--model_name', type=str, default='LightGCN',
                        help='Name of the model')
    parser.add_argument('--embed_dim', type=int, default=32,
                        help='Dimension of embedding vectors')
    parser.add_argument('--n_layers', type=int, default=2,
                        help='Number of GCN layers')

    parser.add_argument('--dataset', type=str, default='LastFM',
                        choices=['LastFM', 'MovieLens'], help='Dataset to use')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the dataset directory')
    parser.add_argument('--num_negatives', type=int, default=1,
                        help='Number of negative samples per positive')
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='Batch size for training')

    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--eval_freq', type=int, default=10,
                        help='Evaluation frequency in epochs')

    parser.add_argument('--topN', nargs='+', type=int, default=[10, 20],
                        help='List of Top-N cutoff values for evaluation')
    parser.add_argument('--reg', type=float, default=1e-5,
                        help='L2 regularization weight')

    parser.add_argument('--save_path', type=str, default=None,
                        help='Optional path to save the model')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    opts = parse_args()
    manager = LightGCNManager(opts)
    manager.train()
    manager.test()
