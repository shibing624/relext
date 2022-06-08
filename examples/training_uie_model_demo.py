# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import argparse
import sys

sys.path.append('..')
from relext.finetune import do_train
from relext.export_model import export_to_static_graph

if __name__ == "__main__":
    # yapf: disable
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", default=16, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--train_path", default=None, type=str, help="The path of train set.")
    parser.add_argument("--dev_path", default=None, type=str, help="The path of dev set.")
    parser.add_argument("--save_dir", default='./checkpoint/', type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--save_static_model_dir", default='./checkpoint/static/', type=str,
                        help="The path of model parameter in static graph to be saved.")
    parser.add_argument("--max_seq_len", default=512, type=int, help="The maximum input sequence length. "
                                                                     "Sequences longer than this will be split automatically.")
    parser.add_argument("--num_epochs", default=100, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--seed", default=1000, type=int, help="Random seed for initialization")
    parser.add_argument("--logging_steps", default=10, type=int, help="The interval steps to logging.")
    parser.add_argument("--valid_steps", default=100, type=int,
                        help="The interval steps to evaluate model performance.")
    parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu",
                        help="Select which device to train model, defaults to gpu.")
    parser.add_argument("--model", choices=["uie-base", "uie-tiny"], default="uie-base", type=str,
                        help="Select the pretrained model for few-shot learning.")
    parser.add_argument("--init_from_ckpt", default=None, type=str,
                        help="The path of model parameters for initialization.")

    args = parser.parse_args()
    # yapf: enable

    print(args)
    do_train(**vars(args))
    export_to_static_graph(args.save_dir, args.save_static_model_dir)
