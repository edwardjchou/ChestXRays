"""This file contains the options used by the training procedure. All options must be passed via the command line."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse


def parser():
    """Adds various options to ArgParse (library used to parse command line arguments).

    Returns:
        opt: An ArgParse dictionary containing keys (argument name) and values (argument value).
    """
    p = argparse.ArgumentParser()
    p.add_argument('--seed', default=1988, type=int, help='Random seed. Use -1 to enable random start.')

    # CPU/GPU and logging settings.
    p.add_argument('--tb_interval', default=100, help='Write TensorBoard summaries to disk every X iterations.')
    p.add_argument('--val_interval', default=5, help='Interval (in epochs) to run the validation set.')
    p.add_argument('--log_loss', default=False, help='Boolean determining if the loss is log or not')

    # Model settings.
    p.add_argument('--run_name', default='', type=str, help='name of the run for saving')
    p.add_argument('--train_input_file', type=str, help='Training dataset',
                   default='/data/dhyun/simulation/despeckle_L12-3v/train_28x28_40k.mat')
    p.add_argument('--train_labels_file', type=str, help='Training dataset',
                   default='/data/dhyun/simulation/despeckle_L12-3v/train_28x28_40k.mat')
    p.add_argument('--val_dataset', type=str, help='Validation dataset',
                   default='/data/dhyun/simulation/despeckle_L12-3v/train_28x28_40k.mat')
    p.add_argument('--test_dataset', type=str, help='Test dataset',
                   default='/data/dhyun/simulation/despeckle_L12-3v/train_28x28_40k.mat')
    p.add_argument('--n_train', default=0, type=int, help='Number of training samples to use')
    p.add_argument('--n_val', default=1000, type=int, help='Number of validation samples to use')
    p.add_argument('--save_dir', type=str, help='output data directory',
					default ='/home/llbricks/output_data/')
    p.add_argument('--arch', default = '3d', type=str, help='Network architecture to use.')
    p.add_argument('--use_channel_data', default=True, type=bool, help='Use channel data input')

    # General optimization settings.
    p.add_argument('--batch_size', default=64 , type=int, help='Batch size.')
    p.add_argument('--dropout', type=float, help='define the percentage of dropout to apply', default = 0.6)
    p.add_argument('--reg1', default=0.0001, type=float, help='L1 reg scalar')
    p.add_argument('--reg2', default=0.0001, type=float, help='L2 reg scalar')
    p.add_argument('--lr', default=2e-5, type=float, help='Initial learning rate.')
    p.add_argument('--lr_decay_rate', default=0.95, type=float, help='Learning rate decay rate.')
    p.add_argument('--lr_decay_epoch', default=10, type=float, help='Decay the learning rate after these many epochs.')
    p.add_argument('--n_epochs', default=50, type=int, help='Max number of training epochs.')
    p.add_argument('--n_filts', default=8, type=int, help='Number of filters in beamform network')
    p.add_argument('--momentum', default=0.9, type=float, help='Momentum for the optimizer.')

    # Inference mode settings
    p.add_argument('--model_ckpt', default='', type=str, help='Location of model file (for inference).')
    p.add_argument('--input_file', default='', type=str, help='Location of input data to be used for predictions.')
    p.add_argument('--pred_file', default='', type=str, help='Location to place predicted results.')

    # Retrain mode settings
    p.add_argument('--input_model_ckpt', default='', type=str, help='Location of input model file (for inference).')
    p.add_argument('--freeze_weights', default='False', type=bool, help='Prevent network from updating loaded weights')

    opt = p.parse_args()
    return opt
