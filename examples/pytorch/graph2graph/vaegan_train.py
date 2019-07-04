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
parser.add_argument("--epoch", dest="epoch", type=int, default=100, help="num of epoch max")
parser.add_argument("--load_epoch", dest="load_epoch", type=int, default=-1, help="load param's epoch num")
parser.add_argument("--batch", dest="batch_size", type=int, default=40)
parser.add_argument("--hidden", dest="hidden_size", type=int, default=200)
parser.add_argument("--latent", dest="latent_size", type=int, default=56)
parser.add_argument("--depth", dest="depth", type=int, default=3)
parser.add_argument("--beta", dest="beta", type=int, default=1.0)

# Ported from arae_train.py, not sure what are these
parser.add_argument("--diter", dest="diter", type=int, default=5, help="descriminator num of iteration")
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
#TODO(hq): take care of the replicate arg
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

sample = main_dataset[0][0]
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

MAX_EPOCH = opts.epoch
PRINT_ITER = 1

param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))

assert opts.gan_batch_size <= opts.batch_size
num_epoch = (opts.epoch - opts.load_epoch - 1) * (opts.diter + 1) * 10

x_dataloader = DataLoader(
    x_dataset,
    batch_size=opts.gan_batch_size,
    shuffle=True,
    num_workers=0,
    collate_fn=Graph2GraphCollator(vocab, training=False, mode="pair"),
    drop_last=True,
    worker_init_fn=worker_init_fn
)
y_dataloader = DataLoader(
    y_dataset,
    batch_size=opts.gan_batch_size,
    shuffle=True,
    num_workers=0,
    collate_fn=Graph2GraphCollator(vocab, training=False, mode='single'),
    drop_last=True,
    worker_init_fn=worker_init_fn
)
main_dataloader = DataLoader(
    main_dataset,
    batch_size=opts.gan_batch_size,
    shuffle=True,
    num_workers=0,
    collate_fn=Graph2GraphCollator(vocab, training=True, mode="pair"),
    drop_last=True,
    worker_init_fn=worker_init_fn
)
x_dataloader = iter(x_dataloader)
y_datloader = iter(y_dataloader)

for epoch in range(opts.load_epoch + 1, MAX_EPOCH):
    for it, batch in enumerate(main_dataloader):
        
        #1. Train encoder & decoder
        model.zero_grad()
        topology_ce, label_ce, assm_loss, kl_div, topo_acc, label_acc, assm_acc = model(batch)
        loss = topology_ce + label_ce + assm_loss + kl_div
        print('iteration %d | topology %.3f | label %.3f | assml %.3f | kl %.3f | %.3f' % (it, topology_ce.item(), label_ce.item(), assm_loss.item(), kl_div.item(), loss.item()))
        print("accuracy: topology %.3f | label %.3f | assm %.3f"%(topo_acc, label_acc, assm_acc))
        print("++++++++++++++++++++++++++++++++++++++\n")
        nn.utils.clip_grad_norm_(model.parameters(), opts.clip_norm)
        optimizer.step()

        #2. Train discriminator
        for i in range(opts.diter):
            GAN.netD.zero_grad()
            x_batch = next(x_dataloader)
            y_batch = next(y_dataloader)
            # Notice that x_batch still contains an unnecessary y
            d_loss, gp_loss = GAN.train_D(x_batch, y_batch, model)
            optimizerD.step()
        
        #3. Train generator(ARAE fashion)
        model.zero_grad()
        GAN.zero_grad()
        x_batch, _ = next(x_dataloader)
        y_batch = next(y_dataloader)
        g_loss = GAN.train_G(x_batch, y_batch, model)
        nn.utils.clip_grad_norm(model.parameters(), opts.clip_norm)
        optimizerG.step()
        print("Disc: %.3f, Gen: %.4f, GP: %.4f, PNorm: %.2f, %.2f, GNorm: %.2f, %2f" %
        (d_loss, g_loss, gp_loss, param_norm(model), param_norm(GAN.netD), grad_norm(model), grad_norm(GAN.netD)))

         if (it + 1) % 1500 == 0: #Fast annealing
            scheduler.step()
            print("learning rate: %.6f" % scheduler.get_lr()[0])
            print("saving model")
            torch.save(model.state_dict(),
                       cur_dir + opts.save_dir + opts.experiment + "/model.iter-%d-%d" % (epoch, it + 1))
            torch.save(gan.state_dict(),
                       cur_dir + opts.save_dir + opts.experiment + "/model.iter-%d-%d" % (epoch, it + 1) + "-gan")


    scheduler.step()
    print("EPOCH ", epoch)
    print("learning rate: %.6f" % scheduler.get_lr()[0])
    torch.save(model.state_dict(), cur_dir + opts.save_dir + opts.experiment + "/model.iter-" + str(epoch))
    torch.save(GAN.state_dict(), cur_dir + opts.save_dir + opts.experiment + "/model.iter-" + str(epoch)+"-gan")