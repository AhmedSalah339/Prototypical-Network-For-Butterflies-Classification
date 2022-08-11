# coding=utf-8
import os
import argparse

def get_classes_means_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-hyb', '--hybrid_model_weights',
                        type=str,
                        help='path to hyprid model weights')
    parser.add_argument('-pro', '--proto_weights',
                        type=str,
                        help='path to the protonet weights')
    parser.add_argument('-enc', '--enc_weights',
                        type=str,
                        help='path to the ResNet weights')
    parser.add_argument('-spl', '--split_path',
                        type=str,
                        default='configs'+os.sep+'splits'+os.sep+'split_dict_hybrid_clust.pkl',
                        help='path to the classes - paths dictionary to be used in calculating the mean')
    parser.add_argument('-b_size', '--batch_size',
                        default=32,
                        type=int,
                        help='batch size')
    parser.add_argument('-svp', '--save_path',
                        type=str,
                        default=os.path.join('configs','means.pkl'),
                        help='path to save the mean of the classes')
    
    parser.add_argument('--freeze_encoder',
                        action='store_true',
                        help='freeze the encoder while training')
    parser.add_argument('--freeze_proto',
                        action='store_true',
                        help='freeze the proto while training')
    parser.add_argument('--no_precache',
                        action='store_true',
                        help='not using the precache')
    parser.add_argument('--use_median',
                        action='store_true',
                        help='calculate the median instead of the mean')
    return parser
    