import torch
import dgl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

import math, random, sys, pickle, os
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
parser.add_argument("--ymols", dest="ymols", type=str, default="target", help="target molecule file")
parser.add_argument("--vocab", dest="vocab", type=str, default='vocab', help='Vocab file name')
parser.add_argument("--exp", dest="experiment", type=str, default="qed", help="which experiment")
parser.add_argument("--save_dir", dest="save_dir", type=str, default="/models/", help="save_path")
parser.add_argument("--model", dest="model_path", type=str, default=None)
parser.add_argument("--batch", dest="batch_size", type=int, default=40)
parser.add_argument("--hidden", dest="hidden_size", type=int, default=200)
parser.add_argument("--latent", dest="latent_size", type=int, default=56)
parser.add_argument("--depth", dest="depth", type=int, default=3)
parser.add_argument("--beta", dest="beta", type=int, default=1.0)

# Ported from arae_train.py, not sure what are these
parser.add_argument("--disc_hidden", dest="disc_hidden", type=int, default=300, help="TODO")
parser.add_argument("--gan_batch_size", dest="gan_batch_size", type=int, default=10, help="gan_batch_size")
parser.add_argument("gumbel", action="store_true")
parser.add_argument("--gan_lrG", dest="gan_lrG", type=float, default=1e-4)
parser.add_argument("--gan_lrD", dest="gan_lrD", type=float, default=1e-4)
parser.add_argument("--kl_lambda", type=float, default=1.0)
parser.add_argument("--clip_norm", type=float, default=50.0)
parser.add_argument("--anneal_rate", type=float, default=0.9)


parser.add_argument("--lr", dest="lr", type=float, default=1e-3)
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

parser.add_argument('--gpu', type=int, default=0)

opts = parser.parse_args()
print(opts)

print("loading dataset!")
x_dataset = Graph2GraphDataset(data=opts.train, vocab=opts.vocab, training=False, exp=opts.experiment, mode="pair")
# \TODO is there any one-to-one correspondence between x_dataset?
y_dataset = Graph2GraphDataset(data=opts.ymols, vocab=opts.vocab, training=False, exp=opts.experiment, mode='single')
main_dataset = Graph2GraphDataset(data=opts.train, vocab=opts.vocab, training=True, exp=opts.experiment, mode="pair")
print('dataset loading finished')
vocab = x_dataset.vocab

# TODO(hq): remove
"""
batch_size = int(opts.batch_size)
hidden_size = int(opts.hidden_size)
latent_size = int(opts.latent_size)
depth = int(opts.depth)
beta = float(opts.beta)
"""
lr = float(opts.lr)
lrG = float(opts.gan_lrG)
lrD = float(opts.gan_lrD)
#

sample = dataset[0][0]
args = opts
args.d_ndataG = sample['atom_x_enc'].size(1)
args.d_edataG = sample['bond_x_enc'].size(1)
# input molecular node feature dimension is different between encoding stage
# and decoding stage. Decoding stage's feature does not include Chiral information
args.d_ndataG_dec = sample['atom_x_dec'].size(1)
args.d_edataG_dec = sample['bond_x_dec'].size(1)
args.vocab_size = vocab.size()
# args.d_edataT = 0
model = Graph2Graph(args, vocab)
GAN = ScaffoldGAN(model, opts.disc_hidden, beta=opts.beta, gumbel=opts.gumbel)
cur_dir = os.getcwd()
pickle.dump(args, open(cur_dir + args.save_dir + args.experiment + "/model_args.pickle", "wb"))
#

if opts.model_path is not None:
    param = cur_dir + opts.save_dir + opts.experiment + "/" + opts.model_path
    gan_param = param + "-gan"
    model.load_state_dict(torch.load(param))
    GAN.load_state_dict(torch.load(gan_param))
else:
    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant(param, 0)
        else:
            nn.init.xavier_normal(param)
    
    for param in GAN.parameters():
        if param.dim() == 1:
            nn.init.constant(param, 0)
        else:
            nn.init.xavier_normal(param)


device = torch.device('cpu') if args.gpu < 0 else torch.device('cuda:%d' % args.gpu)
model = model.to(device)
GAN = GAN.to(device)
print("Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,))
print("Scaffold GAN Model #Params: %dK" % (sum([x.nelement() for x in GAN.parameters()]) / 1000,))


optimizer = optim.Adam(model.parameters(), lr=lr)
optimizerG = optim.Adam(model.parameters(), lr=lrG, betas=(0,0.9))
optimizerD = optim.Adam(GAN.netD.parameters(), lr=lrD, betas=(0, 0.9))

scheduler = lr_scheduler.ExponentialLR(optimizer, opts.anneal_rate)
scheduler.step()

MAX_EPOCH = 100
PRINT_ITER = 1

param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))

assert opts.gan_batch_size <= opts.batch_size




# WIPPPPPPPPPP
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
            # X_G, X_T = process(batch[0])
            # Y_G, Y_T = process(batch[1])
            topology_ce, label_ce, assm_loss, kl_div, topo_acc, label_acc, assm_acc = model(batch)
            print("#### This is model embedding")
            print(model.embeddings)
            print("########")
            loss = topology_ce + label_ce + assm_loss + kl_div
            print('iteration %d | topology %.3f | label %.3f | assml %.3f | kl %.3f | %.3f' % (it, topology_ce.item(), label_ce.item(), assm_loss.item(), kl_div.item(), loss.item()))
            print("accuracy: topology %.3f | label %.3f | assm %.3f"%(topo_acc, label_acc, assm_acc))
            print("++++++++++++++++++++++++++++++++++++++\n")
            # Manually toggle on/off assm loss
            #loss = topology_ce + label_ce + kl_div
            #print('topology %.3f | label %.3f | kl %.3f | %.3f' % (topology_ce.item(), label_ce.item(), kl_div.item(), loss.item()))
            
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
                print("saving model")
                torch.save(model.state_dict(),
                           cur_dir + opts.save_dir + opts.experiment + "/model.iter-%d-%d" % (epoch, it + 1))

        scheduler.step()
        print("EPOCH ", epoch)
        print("learning rate: %.6f" % scheduler.get_lr()[0])
        torch.save(model.state_dict(), cur_dir + opts.save_dir + opts.experiment + "/model.iter-" + str(epoch))