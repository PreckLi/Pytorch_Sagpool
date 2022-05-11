import argparse
import torch
from train import train

parser=argparse.ArgumentParser()
parser.add_argument('--seed',type=int,default=666,help='seed')
parser.add_argument('--batch_size',type=int,default=128,help='batch_size')
parser.add_argument('--lr',type=float,default=0.0005,help='lr')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight_decay')
parser.add_argument('--num_hid',type=int,default=128,help='num_hid')
parser.add_argument('--pooling_ratio',type=float,default=0.5,help='pooling_ratio')
parser.add_argument('--dropout',type=float,default=0.5,help='dropout')
parser.add_argument('--dataset',type=str,default='DD',help='TUDataset-DD-proteins')
parser.add_argument('--epochs',type=int,default=1000,help='epochs')
parser.add_argument('--patience',type=int,default=50,help='patience for early stop')
parser.add_argument('--pooling_layer_type', type=str, default='GCNConv',help='pooling_layer_type')

args=parser.parse_args()



if __name__ == '__main__':
    train(args)