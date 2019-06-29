import os
import pickle

import dgl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import math, random, sys
from argparse import ArgumentParser
from collections import deque
import rdkit

from jtnn import *

torch.multiprocessing.set_sharing_strategy('file_system')

def worker_init_fn(id_):
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)
worker_init_fn(None)

parser = ArgumentParser()
parser.add_argument("--data", dest="data", type=str, default="valid", help="validation file name")
parser.add_argument("--vocab", dest="vocab", type="str", default="vocab", help="vocab dir")
parser.add_argument("--exp", dest="experiment", type=str, default="logp04", help="which experiment")
parser.add_argument("--save_dir", dest="m_dir", default="/models/", help="model param save directory")
parser.add_argument("--model", dest="model_path", type=str, default=None)
parser.add_argument("--hidden", dest="hidden", type=int, default=200)
parser.add_argument("--latent", dest="latent_size", type=int, default=56)
parser.add_argument("--depth", dest="depth", type=int, default=3, help="message passing depth")
parser.add_argument("--beta", dest="beta", type=int, default=1.0)
parser.add_argument("--num_decode", dest="num_decode", type=int, default=20)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--batch_size", dest='batch_size', type=int, default=1)

parser.add_argument('--gpu', type=int, default=0)

args = parser.parse_args()
assert args.batch_size == 1, "evaluation does not support batch decoding!"
print(args)

dataset = Graph2GraphDataset(data=args.data, vocab=args.vocab, training=False, exp=args.experiment, mode="single")
print("loading dataset!")
vocab = dataset.vocab
collate_fn = Graph2GraphCollator(vocab, training=False, mode='single')


#
cur_dir = os.getcwd()
model_args = pickle.load(open(cur_dir + args.save_dir + args.experiment + "/model_args.pickle", "rb"))
model = Graph2Graph(model_args, vocab)
params = cur_dir + args.save_dir + args.experiment + args.model
model.load_state_dict(torch.load(params))
#

#
device = torch.device('cpu') if args.gpu < 0 else torch.device('cuda:%d' % args.gpu)
model = model.to(device)
print("Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,))
#

dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn,
    drop_lst=True,
    worker_init_fn=worker_init_fn
)

torch.manual_seed(args.seed)
with open(cur_dir + "/exp_temp/" + args.experiment + "_temp.txt", 'w') as file:
    for it, batch in enumerate(dataloader):
        gt_smiles = batch[0]['mol_trees'][0].smiles
        X_G, X_T = model.process(batch[0], train=False)
        model.encoder(X_G, X_T)

        tree_vec, mol_vec = X_T.ndata['x'], X_G.ndata['x']

        for i in range(args.num_decode):
            tree_vec, mol_vec = model.sample_with_noise(tree_vec, mol_vec)
            smiles = model.decode(tree_vec, mol_vec)
            file.write(gt_smiles + "," + smiles)