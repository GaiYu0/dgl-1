import dgl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
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
parser.add_argument("--train", dest="train", type=str, default='train_pairs', help='Training file name')
parser.add_argument("--vocab", dest="vocab", type=str, default='vocab', help='Vocab file name')
parser.add_argument("--exp", dest="experiment", type=str, default="logp04", help="which experiment")
parser.add_argument("--save_dir", type=str, dest="save_path")
parser.add_argument("--model", dest="model_path", type=str, default=None)
parser.add_argument("--batch", dest="batch_size", type=int, default=40)
parser.add_argument("--hidden", dest="hidden_size", type=int, default=200)
parser.add_argument("--latent", dest="latent_size", type=int, default=56)
parser.add_argument("--depth", dest="depth", type=int, default=3)
parser.add_argument("--beta", dest="beta", type=int, default=1.0)
parser.add_argument("--lr", dest="lr", type=int, default=1e-5)
parser.add_argument("--test", dest="test", action="store_true")

#
parser.add_argument('--d_msgG', type=int, default=16)
parser.add_argument('--d_msgT', type=int, default=16)
parser.add_argument('--d_ndataT', type=int, default=16)
parser.add_argument('--d_h', type=int, default=16)
parser.add_argument('--d_ud', type=int, default=16)
parser.add_argument('--d_ul', type=int, default=16)
parser.add_argument('--d_xG', type=int, default=16)
parser.add_argument('--d_xT', type=int, default=16)
parser.add_argument('--d_zG', type=int, default=16)
parser.add_argument('--d_zT', type=int, default=16)
parser.add_argument('--n_itersG', type=int, default=4)
parser.add_argument('--n_itersT', type=int, default=4)

parser.add_argument('--gpu', type=int, default=-1)
#

opts = parser.parse_args()
print(opts)

dataset = Graph2GraphDataset(data=opts.train, vocab=opts.vocab, training=True, exp="logp04", mode="pair")
print("loading dataset!")
vocab = dataset.vocab

# TODO(hq): remove
batch_size = int(opts.batch_size)
hidden_size = int(opts.hidden_size)
latent_size = int(opts.latent_size)
depth = int(opts.depth)
beta = float(opts.beta)
lr = float(opts.lr)

#
sample = dataset[0][0]
args = opts
args.d_ndataG = sample['atom_x_enc'].size(1)
args.d_edataG = sample['bond_x_enc'].size(1)
args.vocab_size = vocab.size()
# args.d_edataT = 0
model = Graph2Graph(args)
#

if opts.model_path is not None:
    model.load_state_dict(torch.load(opts.model_path))
else:
    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant(param, 0)
        else:
            nn.init.xavier_normal(param)

device = torch.device('cpu') if args.gpu < 0 else th.device('cuda:%d' % args.gpu)
model = model.to(device)
print("Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,))

optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.ExponentialLR(optimizer, 0.9)
scheduler.step()

MAX_EPOCH = 100
PRINT_ITER = 1

#
def process(batch):
    G = batch['mol_graph_batch']
    G.ndata['f'] = G.ndata['x'].to(device)
    G.pop_n_repr('x')
    # G.pop_n_repr('src_x')
    G.edata['f'] = G.edata['x'].to(device)
    G.pop_e_repr('x')
    for tree in batch['mol_trees']:
        n = tree.number_of_nodes()
        tree.ndata['wid'] = tree.ndata['wid'].to(device)
        tree.ndata['id'] = torch.zeros(n, vocab.size(), device=device)
        tree.ndata['id'][torch.arange(n), tree.ndata['wid']] = 1
    T = dgl.batch(batch['mol_trees'])
    return G, T
#

def train():
    dataset.training = True
    dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=Graph2GraphCollator(vocab, True, mode="pair"),
            drop_last=True,
            worker_init_fn=worker_init_fn)

    for epoch in range(MAX_EPOCH):
        word_acc,topo_acc,assm_acc,steo_acc = 0,0,0,0

        for it, batch in enumerate(dataloader):
            model.zero_grad()
            '''
            try:
                loss, kl_div, wacc, tacc, sacc, dacc = model(batch, beta)
            except:
                print([t.smiles for t in batch['mol_trees']])
                raise
            '''
            #
            X_G, X_T = process(batch[0])
            Y_G, Y_T = process(batch[1])
            topology_ce, label_ce, kl_div = model(X_G, X_T, Y_G, Y_T)
            loss = topology_ce + label_ce + kl_div
            print(topology_ce, label_ce, kl_div, loss)
            #
            loss.backward()
            optimizer.step()

            '''
            word_acc += wacc
            topo_acc += tacc
            assm_acc += sacc
            steo_acc += dacc

            if (it + 1) % PRINT_ITER == 0:
                word_acc = word_acc / PRINT_ITER * 100
                topo_acc = topo_acc / PRINT_ITER * 100
                assm_acc = assm_acc / PRINT_ITER * 100
                steo_acc = steo_acc / PRINT_ITER * 100

                print("KL: %.1f, Word: %.2f, Topo: %.2f, Assm: %.2f, Steo: %.2f, Loss: %.6f" % (
                    kl_div, word_acc, topo_acc, assm_acc, steo_acc, loss.item()))
                word_acc,topo_acc,assm_acc,steo_acc = 0,0,0,0
                sys.stdout.flush()
            '''

            if (it + 1) % 1500 == 0: #Fast annealing
                scheduler.step()
                print("learning rate: %.6f" % scheduler.get_lr()[0])
                torch.save(model.state_dict(),
                           opts.save_path + "/model.iter-%d-%d" % (epoch, it + 1))

        scheduler.step()
        print("learning rate: %.6f" % scheduler.get_lr()[0])
        torch.save(model.state_dict(), opts.save_path + "/model.iter-" + str(epoch))

def test():
    dataset.training = False
    dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=Graph2GraphCollator(vocab, False),
            drop_last=True,
            worker_init_fn=worker_init_fn)

    # Just an example of molecule decoding; in reality you may want to sample
    # tree and molecule vectors.
    for it, batch in enumerate(dataloader):
        gt_smiles = batch['mol_trees'][0].smiles
        print(gt_smiles)
        model.move_to_cuda(batch)
        _, tree_vec, mol_vec = model.encode(batch)
        tree_vec, mol_vec, _, _ = model.sample(tree_vec, mol_vec)
        smiles = model.decode(tree_vec, mol_vec)
        print(smiles)

if __name__ == '__main__':
    if opts.test:
        test()
    else:
        train()

    print('# passes:', model.n_passes)
    print('Total # nodes processed:', model.n_nodes_total)
    print('Total # edges processed:', model.n_edges_total)
    print('Total # tree nodes processed:', model.n_tree_nodes_total)
    print('Graph decoder: # passes:', model.jtmpn.n_passes)
    print('Graph decoder: Total # candidates processed:', model.jtmpn.n_samples_total)
    print('Graph decoder: Total # nodes processed:', model.jtmpn.n_nodes_total)
    print('Graph decoder: Total # edges processed:', model.jtmpn.n_edges_total)
    print('Graph encoder: # passes:', model.mpn.n_passes)
    print('Graph encoder: Total # candidates processed:', model.mpn.n_samples_total)
    print('Graph encoder: Total # nodes processed:', model.mpn.n_nodes_total)
    print('Graph encoder: Total # edges processed:', model.mpn.n_edges_total)
