#!/usr/bin/env python
#-*- coding:utf-8 _*-

import argparse






def get_args():
    parser = argparse.ArgumentParser(description='GNOT for operator learning')
    parser.add_argument('--dataset',type=str,
                        default='ns2d',
                        choices = ['micro2d_sens_100x12', 'micro2d_sens_30x40', 'shape2d_sens', 'inductor2d_sens', 'drone3d_sens'])


    parser.add_argument('--component',type=str,
                        default='all',)



    parser.add_argument('--seed', type=int, default=2023, metavar='Seed',
                        help='random seed (default: 1127802)')

    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--use-tb', type=int, default=0, help='whether use tensorboard')
    parser.add_argument('--comment',type=str,default="",help="comment for the experiment")

    parser.add_argument('--train-num', type=str, default='all')
    parser.add_argument('--test-num', type=str, default='all')
    parser.add_argument('--sample-rate', type=int, default=1, choices=[1, 2, 4, 8, 16])

    parser.add_argument('--sort-data',type=int, default=0)

    parser.add_argument('--normalize_x', type=str, default='unit',
                        choices=['none', 'minmax', 'unit'])
    parser.add_argument('--use-normalizer', type=str, default='unit',
                        choices=['none', 'minmax', 'unit', 'quantile', 'log'],
                        help="whether normalize y")


    parser.add_argument('--epochs', type=int, default=500, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--optimizer', type=str, default='AdamW',choices=['Adam','AdamW'])

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='max learning rate (default: 0.001)')
    parser.add_argument('--weight-decay',type=float,default=5e-6
                        )
    parser.add_argument('--grad-clip', type=str, default=1000.0
                        )
    parser.add_argument('--batch-size', type=int, default=4, metavar='bsz',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--val-batch-size', type=int, default=8, metavar='bsz',
                        help='input batch size for validation (default: 4)')


    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')


    parser.add_argument('--lr-method',type=str, default='cycle',
                        choices=['cycle','step','warmup'])
    parser.add_argument('--lr-step-size',type=int, default=50
                        )
    parser.add_argument('--warmup-epochs',type=int, default=50)

    parser.add_argument('--loss-name',type=str, default='rel2',
                        choices=['rel2','rel1', 'l2', 'l1'])
    #### public model architecture parameters

    parser.add_argument('--model-name', type=str, default='GNOT',
                        choices=['CGPT', 'GNOT', 'RNO', 'FNO', 'MIONet', 'RFNO', 'RGT', 'RGNOT'])
    parser.add_argument('--n-hidden',type=int, default=64)
    parser.add_argument('--n-layers',type=int, default=3)

    #### MLP parameters

    # common
    parser.add_argument('--act', type=str, default='gelu',choices=['gelu','relu','tanh','sigmoid'])
    parser.add_argument('--n-head',type=int, default=1)
    parser.add_argument('--ffn-dropout', type=float, default=0.0, metavar='ffn_dropout',
                        help='dropout for the FFN in attention (default: 0.0)')
    parser.add_argument('--attn-dropout',type=float, default=0.0)
    parser.add_argument('--mlp-layers',type=int, default=3)

    # Transformer
    # parser.add_argument('--subsampled-len',type=int, default=256)
    parser.add_argument('--attn-type',type=str, default='linear', choices=['virtualfourier', 'physicsfourier', 'physics', 'nonlinear','linear'])
    parser.add_argument('--hfourier-dim',type=int,default=0)

    # Physics
    parser.add_argument('--n-slice',type=int, default=128)
    parser.add_argument('--modes',type=int, default=8)

    # GNOT
    parser.add_argument('--n-experts',type=int, default=1)
    parser.add_argument('--branch-sizes',nargs="*",type=int, default=[2])
    parser.add_argument('--n-inner',type=int, default=4)

    # RNO
    parser.add_argument('--gamma', type=float, default=0.3)
    parser.add_argument('--noref', type=float, default=0.3)

    # Sobolev Training
    parser.add_argument('--sobolev', action='store_true', default=False,
                        help='enables Sobolev training')
    parser.add_argument('--lamb', type=float, default=0.3)

    return parser.parse_args()

