# coding=utf-8
import os
import argparse


def get_train_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-sp','--split_path',
                        type=str,
                        help='path to the split dictionary',
                        default='configs'+os.sep+'splits'+os.sep+'split_dict_hybrid_clust.pkl')
    parser.add_argument('-exp', '--experiment_root',
                        type=str,
                        help='root where to store models, losses and accuracies',
                        default='exp')

    parser.add_argument('-nep', '--epochs',
                        type=int,
                        help='number of epochs to train for',
                        default=100)

    parser.add_argument('-lr', '--learning_rate',
                        type=float,
                        help='learning rate for the model, default=0.001',
                        default=0.001)

    parser.add_argument('-lrS', '--lr_scheduler_step',
                        type=int,
                        help='StepLR learning rate scheduler step, default=20',
                        default=20)

    parser.add_argument('-lrG', '--lr_scheduler_gamma',
                        type=float,
                        help='StepLR learning rate scheduler gamma, default=0.5',
                        default=0.5)

    parser.add_argument('-its', '--iterations',
                        type=int,
                        help='number of episodes per epoch, default=100',
                        default=200)

    parser.add_argument('-cTr', '--classes_per_it_tr',
                        type=int,
                        help='number of random classes per episode for training, default=60',
                        default=10)

    parser.add_argument('-nsTr', '--num_support_tr',
                        type=int,
                        help='number of samples per class to use as support for training, default=5',
                        default=2)

    parser.add_argument('-nqTr', '--num_query_tr',
                        type=int,
                        help='number of samples per class to use as query for training, default=5',
                        default=2)

    parser.add_argument('-cVa', '--classes_per_it_val',
                        type=int,
                        help='number of random classes per episode for validation, default=5',
                        default=2)

    parser.add_argument('-nsVa', '--num_support_val',
                        type=int,
                        help='number of samples per class to use as support for validation, default=5',
                        default=2)

    parser.add_argument('-nqVa', '--num_query_val',
                        type=int,
                        help='number of samples per class to use as query for validation, default=15',
                        default=2)

    parser.add_argument('-seed', '--manual_seed',
                        type=int,
                        help='input for the manual seeds initializations',
                        default=7)
    parser.add_argument('-hyb', '--hybrid_model_weights',
                        type=str,
                        help='path to hypried model weights')
    parser.add_argument('-pro', '--proto_weights',
                        type=str,
                        help='path to the protonet weights')
    parser.add_argument('-enc', '--enc_weights',
                        type=str,
                        help='path to the ResNet weights')
    parser.add_argument('--precache',
                        action='store_true',
                        
                        help='enables precashing the dataset')
    parser.add_argument('--freeze_encoder',
                        action='store_true',
                        help='freeze the encoder while training')
    parser.add_argument('--freeze_proto',
                        action='store_true',
                        help='freeze the proto while training')
    parser.add_argument('--no_precache',
                        action='store_true',
                        help='not using the precache')
    parser.add_argument('--cuda',
                        action='store_true',
                        
                        help='enables cuda')

    return parser
