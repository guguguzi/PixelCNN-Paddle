"""Main training script for models."""

import os
import argparse

import paddle
import pixel_cnn



def main(args):
    pixel_cnn.reproduce(args.n_epochs, args.batch_size, args.logdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_epochs", type=int, help="number of training epochs", default=457
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="the training and evaluation batch_size",
        default=256,
    )
    parser.add_argument(
        "--logdir",
        type=str,
        help="the directory where to log model parameters and TensorBoard metrics",
        default="../ckpts",
    )
    parser.add_argument(
        "--gpus", type=int, help="number of GPUs to run the model on", default=0
    )
    args = parser.parse_args()

    main(args)
