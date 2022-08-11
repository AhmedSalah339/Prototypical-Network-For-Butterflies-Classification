# coding=utf-8
import os
import argparse

def get_predict_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-exp', '--examples_path',
                        type=str,
                        help='path to the examples folder to be predicted')
    parser.add_argument('-suf', '--suffix',
                        type=str,
                        default='*.jpg',
                        help='expression of the files to be predicted ')
    parser.add_argument('-hyb', '--hybrid_model_weights',
                        type=str,
                        help='path to hyprid model weights')
    parser.add_argument('-pro', '--proto_weights',
                        type=str,
                        help='path to the protonet weights')
    parser.add_argument('-enc', '--enc_weights',
                        type=str,
                        help='path to the ResNet weights')
    
    parser.add_argument('-b_size', '--batch_size',
                        default=32,
                        type=int,
                        help='batch size')
    parser.add_argument('-svp', '--save_path',
                        type=str,
                        default='results.csv',
                        help='path to save the results')
    parser.add_argument('-mp', '--means_path',
                        type=str,
                        default='configs'+os.sep+'means.pkl',
                        help='path to the means to be used in prediction')
    
    
    parser.add_argument('--no_precache',
                        action='store_true',
                        help='not using the precache')
    
    return parser
    