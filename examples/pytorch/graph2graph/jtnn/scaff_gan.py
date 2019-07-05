import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

import dgl
import dgl.function as fn

# from mol_tree import Vocab, MolTree
from .g2g_encoder import G2GEncoder
from .graph2graph import process

def copy_src(G, u, uv):
    src, dst = G.edges()
    G.edata[uv] = G.ndata[u][src]
 
def copy_dst(G, v, uv):
    src, dst = G.edges()
    G.edata[uv] = G.ndata[v][dst]


class ScaffoldGAN(nn.Module):
    def __init__(self, g2g, hidden_size, beta, gumbel=False):
        super(ScaffoldGAN, self).__init__()
        self.hidden_size = hidden_size
        self.beta = beta
        self.gumbel = gumbel

        self.netG = Generator(g2g.decoder)
        self.netD = nn.Sequential(
            nn.Linear(g2g.hidden_size * 2, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyRelU(0.1),
            nn.Linear(hidden_size, 1)
        )
    
    def reset_netG(self, g2g):
        self.netG = Generator(g2g.decoder)
    
    def encode_real(self, y_batch, g2g):
        #Generate real y_root features
        y_root_vecs = self.netG(y_batch)
        return y_root_vecs
    
    def encode_fake(self, z_batch, g2g):
        Z_G, Z_T = self.process(z_batch[0], train=False)
        g2g.encoder(Z_G, Z_T)
        pred_root_vecs = []
        Z_GT_pair = list(zip(dgl.unbatch(Z_G), dgl.unbatch(Z_T)))
        for Z_G, Z_T in Z_GT_pair:
            z_G = Z_G.ndata['x']
            z_T = Z_T.ndata['x']
            z_T_tilde, z_G_tilde = g2g.sample_with_noise(z_T, z_G)
            root_vec, _ = g2g.decoder.soft_decode(
                
            )

        # blablabla
        
    
    def train_D(self, x_batch, y_batch, g2g):
        real_vecs = self.encode_real(y_batch, g2g).detach()
        fake_vecs = self.encode_fake(x_batch, g2g).detach()
        real_score = self.netD(real_vecs)
        fake_score = self.nedT(fake_vecs)
        score = fake_score.mean() - real_score.mean() # maximize -> minimize minus
        score.backward()

        # Gradient Penalty
        inter_gp, inter_norm = self.gradient_penalty(real_vecs, fake_vecs)
        inter_gp.backward()
        return -score.item(), inter_norm

    def train_G(self, x_batch, y_batch, g2g):
        real_vecs = self.encode_real(y_batch, g2g)
        fake_vecs = self.encode_fake(x_batch, g2g)
        real_score = self.netD(real_vecs)
        fake_score = self.netD(fake_vecs)
        score = real_score.mean() - fake_score.mean()
        score.backward()
        return score.item()

    def gradient_penalty(self, real_vecs, fake_vecs):
        eps = create_var(th.rand(real_vecs.size(0), 1))
        inter_data = eps * real_vecs + (1 - eps) * fake_vecs
        inter_data = autograd.Variable(inter_data, requires_grad=True)
        inter_score = self.netD(inter_data).squeeze(-1)

        inter_grad = autograd.grad(inter_score, inter_data, 
                grad_outputs=th.ones(inter_score.size()).cuda(),
                create_graph=True, retain_graph=True, only_inputs=True)[0]

        inter_norm = inter_grad.norm(2, dim=1)
        inter_gp = ((inter_norm - 1) ** 2).mean() * self.beta
        #inter_norm = (inter_grad ** 2).sum(dim=1)
        #inter_gp = torch.max(inter_norm - 1, self.zero).mean() * self.beta

        return inter_gp, inter_norm.mean().item()

class Generator(nn.Module):
    
    def __init__(self, g2g_decoder):
        self.hidden_size = g2g_decoder.hidden_size
        self.embeddings = g2g_decoder.embeddings
        self.tree_gru = g2g_decoder.tree_gru

    def forward(self, ybatch, depth):
        Y_G, Y_T = self.process(ybatch, train=False)
        device=Y_G.ndata['f'].device
        Y_T.ndata['f'] = Y_T.ndata['id'] @ self.embeddings

        copy_src(Y_T, 'f', 'f_src')
        copy_dst(Y_T, 'f', 'f_dst')

        # T_lg is constructed from the groundtruth tree Y_T
        T_lg = Y_T.line_graph(backtracking=False, shared=True)
        T_lg.ndata['msg'] = th.zeros(T_lg.number_of_nodes(), self.d_msgT, device=device)

        roots = np.cumsum([0] + Y_T.batch_num_nodes)[:-1]

        for it in range(depth):
            self.tree_gru(T_lg)
        readout_msg_fn = fn.copy_e(e='msg', out='msg')
        readout_reduce_fn = fn.reducer.sum(msg='msg', out='sum_msg')
        readout_apply_fn = lambda nodes : {"r" : th.cat(nodes.data['f'], nodes.data['sum_msg'])}
        Y_T.update_all(readout_msg_fn, readout_reduce_fn, readout_apply_fn)

        return Y_T.nodes[roots].data['r']
        
        




